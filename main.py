#!/usr/bin/env python3
"""
RecurseZero: GPU-Resident Chess RL Agent
Main training script with smart VRAM management.

Features:
- Automatic VRAM cleaning before training
- Auto-tuned batch size based on available memory
- Simple FP32 training (stable and predictable)
"""

print("=" * 60)
print("🧠 RecurseZero - Starting...")
print("=" * 60)
print()

import sys
import time
import subprocess
import gc
import os

# ═══════════════════════════════════════════════════════════════════════════════
# SMART VRAM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu', 
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 5:
                return {
                    'gpu_util': int(parts[0].strip()),
                    'mem_used': int(parts[1].strip()),
                    'mem_total': int(parts[2].strip()),
                    'mem_free': int(parts[3].strip()),
                    'temp': int(parts[4].strip()),
                }
    except Exception:
        pass
    return {'gpu_util': -1, 'mem_used': -1, 'mem_total': -1, 'mem_free': -1, 'temp': -1}

def format_gpu_str(stats):
    if stats['gpu_util'] < 0:
        return ""
    return f"GPU: {stats['gpu_util']}% | VRAM: {stats['mem_used']}/{stats['mem_total']}MB | {stats['temp']}°C"

def clean_vram():
    """Clean up GPU VRAM before training."""
    print("🧹 Cleaning VRAM...", flush=True)
    
    # Force Python garbage collection
    gc.collect()
    
    # Try to clear JAX caches
    try:
        import jax
        jax.clear_caches()
        print("  ✓ JAX caches cleared")
    except Exception:
        pass
    
    # Force garbage collection again
    gc.collect()
    
    stats = get_gpu_stats()
    if stats['mem_free'] > 0:
        print(f"  ✓ Free VRAM: {stats['mem_free']}MB / {stats['mem_total']}MB")
    
    return stats

def auto_batch_size(target_vram_usage=0.7):
    """
    Return optimal batch size for speed.
    
    Testing showed:
    - batch 2048: 2.9 s/s (BEST)
    - batch 3584: 0.9 s/s (slower)
    - batch 4096: 1.4 s/s (slower)
    
    Larger batch = slower per-step, doesn't improve throughput.
    """
    # Fixed optimal batch size based on testing
    OPTIMAL_BATCH = 2048
    print(f"  ✓ Using optimal batch size: {OPTIMAL_BATCH}")
    return OPTIMAL_BATCH

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
    console = Console()
    print("✓ Rich library loaded")
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ Rich not installed - using plain output")

print("Loading JAX...", flush=True)
import jax
import jax.numpy as jnp
print(f"✓ JAX loaded (backend: {jax.default_backend()})", flush=True)

print("Loading Optax...", flush=True)
import optax
print("✓ Optax loaded", flush=True)

print("Loading environment...", flush=True)
from env.pgx_wrapper import RecurseEnv
print("✓ Environment loaded", flush=True)

print("Loading model...", flush=True)
from model.agent import RecurseZeroAgentFast
print("✓ Model loaded", flush=True)

print("Loading training loop...", flush=True)
from training.loop import TrainState, resident_train_step
print("✓ Training loop loaded", flush=True)

print("Loading hardware config...", flush=True)
from optimization.hardware_compat import setup_jax_platform
print("✓ Hardware config loaded", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("INITIALIZATION")
    print("=" * 60)
    
    setup_jax_platform()
    
    # Clean VRAM and auto-tune batch size
    print()
    clean_vram()
    BATCH_SIZE = auto_batch_size(target_vram_usage=0.75)
    print()
    
    # Apply Metal Patch for Pgx
    print("Applying Pgx patches...", flush=True)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    print(f"Initializing environment (batch_size={BATCH_SIZE})...", flush=True)
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    env_state = env.init(env_key)
    print(f"✓ Environment: Batch={BATCH_SIZE}, Actions={env.num_actions}", flush=True)
    
    # 2. Initialize Model (simple FP32 - stable and predictable)
    print("Initializing model...", flush=True)
    agent = RecurseZeroAgentFast(num_actions=env.num_actions)
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    # Count parameters
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model initialized ({param_count:,} parameters)", flush=True)
    
    # 3. Optimizer Setup
    print("Setting up optimizer...", flush=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, weight_decay=0.01)
    )
    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=optimizer
    )
    print("✓ Optimizer ready", flush=True)
    
    # Check VRAM after setup
    gc.collect()
    stats = get_gpu_stats()
    print(f"✓ Setup complete | {format_gpu_str(stats)}")
    
    # 4. JIT Warmup
    print()
    print("=" * 60)
    print("⚙️  JIT COMPILING (this takes 1-3 minutes, please wait...)")
    print("=" * 60)
    print()
    
    key, warmup_key = jax.random.split(key)
    warmup_start = time.time()
    
    print("Compiling XLA kernels...", flush=True)
    train_state, env_state, warmup_metrics = resident_train_step(
        train_state, env_state, warmup_key, agent.apply, env.step, env._init
    )
    jax.block_until_ready(warmup_metrics['total_loss'])
    
    warmup_time = time.time() - warmup_start
    gpu_stats = get_gpu_stats()
    print(f"✓ JIT Compilation complete in {warmup_time:.1f}s")
    print(f"  {format_gpu_str(gpu_stats)}")
    
    # 5. Training Loop
    NUM_STEPS = 10000
    PRINT_EVERY = 100
    
    print()
    print("=" * 60)
    print(f"TRAINING ({NUM_STEPS:,} steps @ batch={BATCH_SIZE})")
    print("=" * 60)
    print()
    
    step_times = []
    total_games = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Training...[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            train_task = progress.add_task("Training", total=NUM_STEPS)
            
            for i in range(NUM_STEPS):
                step_start = time.time()
                
                key, step_key = jax.random.split(key)
                train_state, env_state, metrics = resident_train_step(
                    train_state, env_state, step_key, agent.apply, env.step, env._init
                )
                
                jax.block_until_ready(metrics['total_loss'])
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                total_games += int(metrics['games_finished'])
                total_wins += int(metrics['wins'])
                total_losses += int(metrics['losses'])
                total_draws += int(metrics['draws'])
                
                recent_times = step_times[-50:]
                avg_step_time = sum(recent_times) / len(recent_times)
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                
                progress.update(train_task, advance=1)
                
                if i % PRINT_EVERY == 0 or i == NUM_STEPS - 1:
                    gpu_stats = get_gpu_stats()
                    loss_val = float(metrics['total_loss'])
                    win_prob = float(metrics.get('win_probability', 0.5))
                    confidence = float(metrics.get('confidence', 0.5))
                    
                    progress.console.print(
                        f"  Step {i:5d} │ "
                        f"Loss: {loss_val:7.4f} │ "
                        f"P(win): {win_prob:.1%} │ "
                        f"Games: {total_games:,} (W:{total_wins}/L:{total_losses}/D:{total_draws}) │ "
                        f"{steps_per_sec:.1f} s/s │ "
                        f"{format_gpu_str(gpu_stats)}"
                    )
    else:
        for i in range(NUM_STEPS):
            step_start = time.time()
            
            key, step_key = jax.random.split(key)
            train_state, env_state, metrics = resident_train_step(
                train_state, env_state, step_key, agent.apply, env.step, env._init
            )
            
            jax.block_until_ready(metrics['total_loss'])
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            total_games += int(metrics['games_finished'])
            total_wins += int(metrics['wins'])
            total_losses += int(metrics['losses'])
            total_draws += int(metrics['draws'])
            
            if i % PRINT_EVERY == 0:
                recent_times = step_times[-50:]
                avg_step_time = sum(recent_times) / len(recent_times)
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                print(f"Step {i:5d} | Loss: {float(metrics['total_loss']):.4f} | "
                      f"Games: {total_games} | {steps_per_sec:.1f} s/s")
    
    # 6. Training Complete
    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    
    total_time = sum(step_times)
    avg_speed = NUM_STEPS / total_time if total_time > 0 else 0
    positions_per_sec = avg_speed * BATCH_SIZE
    
    print(f"  Steps: {NUM_STEPS:,} | Time: {total_time:.0f}s | Speed: {avg_speed:.1f} s/s")
    print(f"  Positions/sec: {positions_per_sec:,.0f}")
    print(f"  Games played: {total_games:,}")
    print(f"  W/L/D: {total_wins}/{total_losses}/{total_draws}")
    if total_games > 0:
        print(f"  Win rate: {100*total_wins/total_games:.1f}%")
        print(f"  Draw rate: {100*total_draws/total_games:.1f}%")
    print()

if __name__ == "__main__":
    main()
