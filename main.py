#!/usr/bin/env python3
"""
RecurseZero: GPU-Resident Chess RL Agent
Main training script with live progress monitoring.

Key features:
- BFloat16 mixed precision (2x memory, faster compute)
- 4096 batch size (massive parallelism)
- DEQ with Anderson Acceleration
- Muesli policy optimization (search-free)
"""

print("=" * 60)
print("🧠 RecurseZero - Starting...")
print("=" * 60)
print()

import sys
import time
import subprocess

# Try to import rich for beautiful output
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

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Initialize mixed precision BEFORE importing models!
# This ensures get_dtype() returns BF16 when models are constructed
# ═══════════════════════════════════════════════════════════════════════════════
print("Initializing mixed precision...", flush=True)
from optimization.mixed_precision import init_mixed_precision, get_dtype, is_bf16_enabled
init_mixed_precision(enable=True)
print(f"  Compute dtype: {get_dtype()}", flush=True)

print("Loading Optax...", flush=True)
import optax
print("✓ Optax loaded", flush=True)

print("Loading environment...", flush=True)
from env.pgx_wrapper import RecurseEnv
print("✓ Environment loaded", flush=True)

# NOW import models (after BF16 is initialized)
print("Loading model...", flush=True)
from model.agent import RecurseZeroAgent, RecurseZeroAgentFast
print("✓ Model loaded", flush=True)

print("Loading training loop...", flush=True)
from training.loop import TrainState, resident_train_step
print("✓ Training loop loaded", flush=True)

print("Loading hardware config...", flush=True)
from optimization.hardware_compat import setup_jax_platform
print("✓ Hardware config loaded", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GPU MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                return {
                    'gpu_util': int(parts[0].strip()),
                    'mem_used': int(parts[1].strip()),
                    'mem_total': int(parts[2].strip()),
                    'temp': int(parts[3].strip()),
                }
    except Exception:
        pass
    return {'gpu_util': -1, 'mem_used': -1, 'mem_total': -1, 'temp': -1}

def format_gpu_str(stats):
    if stats['gpu_util'] < 0:
        return ""
    return f"GPU: {stats['gpu_util']}% | VRAM: {stats['mem_used']}/{stats['mem_total']}MB | {stats['temp']}°C"

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("INITIALIZATION")
    print("=" * 60)
    
    setup_jax_platform()
    
    # BF16 is already initialized above
    print(f"✓ Using dtype: {get_dtype()}")
    
    # Apply Metal Patch for Pgx
    print("Applying Pgx patches...", flush=True)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    BATCH_SIZE = 4096  # Large batch for throughput
    print(f"Initializing environment (batch_size={BATCH_SIZE})...", flush=True)
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    env_state = env.init(env_key)
    print(f"✓ Environment: Batch={BATCH_SIZE}, Actions={env.num_actions}", flush=True)
    
    # 2. Initialize Model (BF16 is already set globally)
    print("Initializing model...", flush=True)
    agent = RecurseZeroAgentFast(num_actions=env.num_actions)
    
    # Use the correct dtype for dummy input
    dummy_dtype = get_dtype()
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=dummy_dtype)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    # Verify dtype
    sample_param = jax.tree_util.tree_leaves(params)[0]
    print(f"✓ Model initialized (param dtype: {sample_param.dtype})", flush=True)
    
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
                    confidence = float(metrics.get('confidence', 0.5))
                    
                    progress.console.print(
                        f"  Step {i:5d} │ "
                        f"Loss: {loss_val:7.4f} │ "
                        f"P(win): {win_prob:.1%} │ "
                        f"Conf: {confidence:.0%} │ "
                        f"Games: {total_games:,} (W:{total_wins}/L:{total_losses}/D:{total_draws}) │ "
                        f"{steps_per_sec:.1f} s/s │ "
                        f"{format_gpu_str(gpu_stats)}"
                    )
    else:
        # Plain output fallback
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
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    total_time = sum(step_times)
    avg_speed = NUM_STEPS / total_time if total_time > 0 else 0
    
    print(f"  Total steps: {NUM_STEPS:,}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg speed: {avg_speed:.2f} steps/s")
    print(f"  Compute dtype: {get_dtype()}")
    print(f"  Total games: {total_games:,}")
    print(f"  Wins/Losses/Draws: {total_wins}/{total_losses}/{total_draws}")
    
    win_rate = total_wins / max(1, total_games) * 100
    draw_rate = total_draws / max(1, total_games) * 100
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Draw rate: {draw_rate:.1f}%")
    print()

if __name__ == "__main__":
    main()
