import jax
import jax.numpy as jnp
import optax
from env.pgx_wrapper import RecurseEnv
from model.agent import RecurseZeroAgent
from training.loop import TrainState, resident_train_step
from optimization.hardware_compat import setup_jax_platform
import time
import subprocess
import sys

# Rich imports for beautiful live dashboard
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# GPU MONITORING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_gpu_stats():
    """
    Get GPU statistics using nvidia-smi (works on Colab/CUDA).
    Returns dict with utilization, memory, temperature.
    """
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
    
    # Fallback for non-NVIDIA systems (MPS, etc.)
    return {
        'gpu_util': -1,
        'mem_used': -1,
        'mem_total': -1,
        'temp': -1,
    }

def create_gpu_panel(stats):
    """Create a Rich panel displaying GPU statistics."""
    if stats['gpu_util'] < 0:
        # No GPU stats available
        return Panel("[dim]GPU stats unavailable (Metal/MPS)[/dim]", 
                     title="🔧 GPU", border_style="dim")
    
    util_color = "green" if stats['gpu_util'] > 50 else "yellow" if stats['gpu_util'] > 10 else "red"
    temp_color = "green" if stats['temp'] < 70 else "yellow" if stats['temp'] < 85 else "red"
    
    mem_pct = (stats['mem_used'] / stats['mem_total']) * 100 if stats['mem_total'] > 0 else 0
    mem_color = "green" if mem_pct < 70 else "yellow" if mem_pct < 90 else "red"
    
    text = Text()
    text.append(f"⚡ Utilization: ", style="bold")
    text.append(f"{stats['gpu_util']:3d}%\n", style=util_color)
    text.append(f"💾 VRAM: ", style="bold")
    text.append(f"{stats['mem_used']:,} / {stats['mem_total']:,} MB ", style=mem_color)
    text.append(f"({mem_pct:.1f}%)\n", style=mem_color)
    text.append(f"🌡️  Temp: ", style="bold")
    text.append(f"{stats['temp']}°C", style=temp_color)
    
    return Panel(text, title="🎮 GPU Status", border_style="cyan")

def create_metrics_table(metrics, step, steps_per_sec, total_steps):
    """Create a Rich table displaying training metrics."""
    table = Table(box=box.ROUNDED, border_style="green", title="📊 Training Metrics")
    
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="white", justify="left")
    
    table.add_row("Step", f"{step} / {total_steps}")
    table.add_row("Steps/sec", f"{steps_per_sec:.2f}")
    table.add_row("─" * 15, "─" * 20)
    table.add_row("Total Loss", f"{metrics.get('total_loss', 0):.4f}")
    table.add_row("Policy Loss", f"{metrics.get('pg_loss', 0):.4f}")
    table.add_row("PVE Loss", f"{metrics.get('pve_loss', 0):.4f}")
    table.add_row("─" * 15, "─" * 20)
    table.add_row("Mean Reward", f"{metrics.get('mean_reward', 0):.4f}")
    table.add_row("Policy Entropy", f"{metrics.get('policy_entropy', 0):.2f}")
    
    # Win probability from value (v in [-1, 1] -> P(win) = (v+1)/2)
    mean_val = float(metrics.get('mean_value', 0))
    win_prob = (mean_val + 1) / 2 * 100
    table.add_row("Win Probability", f"{win_prob:.1f}%")
    
    return table

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit(
        "[bold cyan]🧠 RecurseZero[/bold cyan]\n"
        "[dim]GPU-Resident Chess RL Agent with DEQ[/dim]",
        border_style="cyan"
    ))
    
    setup_jax_platform()
    
    # Apply Metal Patch for Pgx (Simulate GPU-Safe Hashing)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    BATCH_SIZE = 2048 
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    # Initialize env state on GPU
    env_state = env.init(env_key)
    console.print(f"[green]✓[/green] Environment: Batch={BATCH_SIZE}, Actions={env.num_actions}")
    
    # 2. Initialize Model
    agent = RecurseZeroAgent(num_actions=env.num_actions)
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    # 3. Optimizer Setup
    optimizer = optax.adamw(learning_rate=3e-4)
    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=optimizer
    )
    console.print("[green]✓[/green] Model and Optimizer initialized")
    
    # 4. JIT Warmup (Critical for avoiding "stuck" appearance)
    console.print()
    with console.status("[bold yellow]⚙️  JIT Compiling XLA kernels (this takes 1-3 min)...[/bold yellow]", spinner="dots"):
        key, warmup_key = jax.random.split(key)
        warmup_start = time.time()
        
        # Run one step to trigger compilation
        train_state, env_state, warmup_metrics = resident_train_step(
            train_state, env_state, warmup_key, agent.apply, env.step
        )
        # Block until compilation is complete
        jax.block_until_ready(warmup_metrics['total_loss'])
        
        warmup_time = time.time() - warmup_start
    
    console.print(f"[green]✓[/green] JIT Compilation complete in [cyan]{warmup_time:.1f}s[/cyan]")
    console.print()
    
    # 5. Training Loop with Live Dashboard
    NUM_STEPS = 50
    step_fn = resident_train_step
    
    # Tracking
    step_times = []
    current_metrics = warmup_metrics
    
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
            train_state, env_state, metrics = step_fn(
                train_state, env_state, step_key, agent.apply, env.step
            )
            
            # Block to get accurate timing
            jax.block_until_ready(metrics['total_loss'])
            step_time = time.time() - step_start
            step_times.append(step_time)
            current_metrics = metrics
            
            # Calculate steps/sec (rolling average of last 10)
            recent_times = step_times[-10:]
            avg_step_time = sum(recent_times) / len(recent_times)
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            
            # Update progress
            progress.update(train_task, advance=1)
            
            # Print metrics periodically
            if i % 10 == 0 or i == NUM_STEPS - 1:
                gpu_stats = get_gpu_stats()
                
                # Build status line
                loss_val = float(metrics['total_loss'])
                reward_val = float(metrics['mean_reward'])
                entropy_val = float(metrics['policy_entropy'])
                
                gpu_str = ""
                if gpu_stats['gpu_util'] >= 0:
                    gpu_str = f" | GPU: {gpu_stats['gpu_util']}% | VRAM: {gpu_stats['mem_used']}/{gpu_stats['mem_total']}MB | {gpu_stats['temp']}°C"
                
                progress.console.print(
                    f"  [dim]Step {i:3d}[/dim] │ "
                    f"Loss: [yellow]{loss_val:7.4f}[/yellow] │ "
                    f"Reward: [cyan]{reward_val:6.3f}[/cyan] │ "
                    f"Entropy: [magenta]{entropy_val:5.2f}[/magenta] │ "
                    f"[green]{steps_per_sec:.1f} steps/s[/green]"
                    f"[dim]{gpu_str}[/dim]"
                )
    
    # Final Summary
    total_time = sum(step_times)
    avg_steps_per_sec = NUM_STEPS / total_time if total_time > 0 else 0
    
    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Training Complete![/bold green]\n\n"
        f"  Steps: {NUM_STEPS}\n"
        f"  Time: {total_time:.1f}s (+ {warmup_time:.1f}s compile)\n"
        f"  Speed: {avg_steps_per_sec:.2f} steps/sec\n"
        f"  Final Loss: {float(current_metrics['total_loss']):.4f}",
        title="📈 Summary",
        border_style="green"
    ))
    
    # Final GPU stats
    gpu_stats = get_gpu_stats()
    if gpu_stats['gpu_util'] >= 0:
        console.print(create_gpu_panel(gpu_stats))

if __name__ == "__main__":
    main()
