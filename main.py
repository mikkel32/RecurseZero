#!/usr/bin/env python3
"""
RecurseZero: GPU-Resident Chess RL Agent

OPTIMIZATIONS BASED ON RESEARCH:
1. XLA_PYTHON_CLIENT_PREALLOCATE=false (on-demand allocation)
2. Reduced memory preallocation  
3. Optimized batch size for throughput
4. Gradient checkpointing for memory efficiency
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Set memory flags BEFORE importing JAX
# JAX preallocates 75% of GPU memory by default
# ═══════════════════════════════════════════════════════════════════════════════
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Allocate on demand
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'   # If preallocating, use 80%

print("=" * 60)
print("🧠 RecurseZero - Starting...")
print("=" * 60)
print()

import sys
import time
import subprocess
import gc

# ═══════════════════════════════════════════════════════════════════════════════
# GPU MONITORING
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
    
    # Show initial VRAM
    stats = get_gpu_stats()
    print(f"Initial VRAM: {stats['mem_used']}/{stats['mem_total']}MB")
    
    # Apply Metal Patch for Pgx
    print("Applying Pgx patches...", flush=True)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    # TESTED: batch 2048 gives best throughput (1.8 s/s × 2048 = 3686 pos/sec)
    # Larger batch is slower per-step, doesn't improve total throughput
    BATCH_SIZE = 2048
    
    # Verify we're on GPU-only (no CPU fallback)
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if 'gpu' not in str(devices[0]).lower() and 'cuda' not in str(devices[0]).lower():
        print("⚠️  WARNING: Not running on GPU! Check JAX installation.")
    else:
        print(f"✓ Confirmed GPU execution: {devices[0]}")
    
    print(f"Initializing environment (batch_size={BATCH_SIZE})...", flush=True)
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    env_state = env.init(env_key)
    print(f"✓ Environment: Batch={BATCH_SIZE}, Actions={env.num_actions}", flush=True)
    
    # Check VRAM after env init
    gc.collect()
    stats = get_gpu_stats()
    print(f"  After env init: {stats['mem_used']}MB used")
    
    # 2. Initialize Model
    print("Initializing model...", flush=True)
    agent = RecurseZeroAgentFast(num_actions=env.num_actions)
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model initialized ({param_count:,} parameters)", flush=True)
    
    # Check VRAM after model init
    stats = get_gpu_stats()
    print(f"  After model init: {stats['mem_used']}MB used")
    
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
    print(f"TRAINING ({NUM_STEPS:,} steps)")
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
    if total_games > 0:
        print(f"  Win rate: {100*total_wins/total_games:.1f}%")
    print()

if __name__ == "__main__":
    main()
