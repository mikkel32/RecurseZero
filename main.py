#!/usr/bin/env python3
"""
RecurseZero: GPU-Resident Chess RL Agent
Main training script with live progress monitoring.

Key improvements:
- 10,000 training steps (from 50)
- Game auto-reset when terminated
- Win/Loss/Draw tracking
- Reduced DEQ iterations for 2x speed
"""

print("=" * 60)
print("🧠 RecurseZero - Starting...")
print("=" * 60)
print()

import sys
import time

# Try to import rich for beautiful output, fallback to plain print
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
from model.agent import RecurseZeroAgent
print("✓ Model loaded", flush=True)

print("Loading training loop...", flush=True)
from training.loop import TrainState, resident_train_step
print("✓ Training loop loaded", flush=True)

print("Loading hardware config...", flush=True)
from optimization.hardware_compat import setup_jax_platform
print("✓ Hardware config loaded", flush=True)

import subprocess

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
    
    # Initialize Int8 Quantization (2-4x speedup on Tensor Cores)
    try:
        from optimization.quantization import init_quantization
        # Enable quantization on CUDA/Tensor Core hardware
        if jax.default_backend() == 'gpu':
            init_quantization(enable=True)
        else:
            print("⚪ Quantization skipped (not on CUDA GPU)")
    except ImportError:
        print("⚠ Quantization module not available")
    
    # Apply Metal Patch for Pgx
    print("Applying Pgx patches...", flush=True)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    BATCH_SIZE = 2048 
    print(f"Initializing environment (batch_size={BATCH_SIZE})...", flush=True)
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    env_state = env.init(env_key)
    print(f"✓ Environment: Batch={BATCH_SIZE}, Actions={env.num_actions}", flush=True)
    
    # 2. Initialize Model
    print("Initializing model...", flush=True)
    agent = RecurseZeroAgent(num_actions=env.num_actions)
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    print("✓ Model initialized", flush=True)
    
    # 3. Optimizer Setup
    print("Setting up optimizer...", flush=True)
    optimizer = optax.adamw(learning_rate=3e-4)
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
    print(f"✓ JIT Compilation complete in {warmup_time:.1f}s", flush=True)
    
    gpu_stats = get_gpu_stats()
    print(f"  {format_gpu_str(gpu_stats)}", flush=True)
    
    # 5. Training Loop - INCREASED TO 10000 STEPS
    NUM_STEPS = 10000
    PRINT_EVERY = 100
    
    step_times = []
    total_games = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    
    print()
    print("=" * 60)
    print(f"TRAINING ({NUM_STEPS:,} steps)")
    print("=" * 60)
    print()
    
    training_start = time.time()
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            
            train_task = progress.add_task("[cyan]Training...", total=NUM_STEPS)
            
            for i in range(NUM_STEPS):
                step_start = time.time()
                
                key, step_key = jax.random.split(key)
                train_state, env_state, metrics = resident_train_step(
                    train_state, env_state, step_key, agent.apply, env.step, env._init
                )
                
                jax.block_until_ready(metrics['total_loss'])
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Accumulate game stats
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
                    reward_val = float(metrics['mean_reward'])
                    entropy_val = float(metrics.get('policy_entropy', 0))
                    games_this_step = int(metrics['games_finished'])
                    
                    win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
                    
                    progress.console.print(
                        f"  Step {i:5d} │ "
                        f"Loss: {loss_val:7.4f} │ "
                        f"Reward: {reward_val:+6.3f} │ "
                        f"Games: {total_games:,} (W:{total_wins}/L:{total_losses}/D:{total_draws}) │ "
                        f"WinRate: {win_rate:.1f}% │ "
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
            
            if i % PRINT_EVERY == 0 or i == NUM_STEPS - 1:
                gpu_stats = get_gpu_stats()
                loss_val = float(metrics['total_loss'])
                reward_val = float(metrics['mean_reward'])
                entropy_val = float(metrics.get('policy_entropy', 0))
                
                recent_times = step_times[-50:]
                avg_step_time = sum(recent_times) / len(recent_times)
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                
                win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
                
                print(
                    f"Step {i:5d}/{NUM_STEPS} │ "
                    f"Loss: {loss_val:7.4f} │ "
                    f"Reward: {reward_val:+6.3f} │ "
                    f"Games: {total_games:,} W:{total_wins}/L:{total_losses}/D:{total_draws} │ "
                    f"WinRate: {win_rate:.1f}% │ "
                    f"{steps_per_sec:.1f} s/s │ "
                    f"{format_gpu_str(gpu_stats)}",
                    flush=True
                )
    
    # Final Summary
    total_training_time = time.time() - training_start
    total_time = sum(step_times)
    avg_steps_per_sec = NUM_STEPS / total_time if total_time > 0 else 0
    positions_per_sec = avg_steps_per_sec * BATCH_SIZE
    
    print()
    print("=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Steps:          {NUM_STEPS:,}")
    print(f"  Time:           {total_training_time:.1f}s (+ {warmup_time:.1f}s compile)")
    print(f"  Speed:          {avg_steps_per_sec:.2f} steps/sec")
    print(f"  Positions/sec:  {positions_per_sec:,.0f}")
    print(f"  Final Loss:     {float(metrics['total_loss']):.4f}")
    print()
    print("  📊 Game Statistics:")
    print(f"     Total Games: {total_games:,}")
    print(f"     Wins:        {total_wins:,} ({total_wins/max(total_games,1)*100:.1f}%)")
    print(f"     Losses:      {total_losses:,} ({total_losses/max(total_games,1)*100:.1f}%)")
    print(f"     Draws:       {total_draws:,} ({total_draws/max(total_games,1)*100:.1f}%)")
    print()
    
    gpu_stats = get_gpu_stats()
    print(f"  {format_gpu_str(gpu_stats)}")
    print()

if __name__ == "__main__":
    main()
