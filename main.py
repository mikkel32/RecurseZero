import jax
import jax.numpy as jnp
import optax
from env.pgx_wrapper import RecurseEnv
from model.agent import RecurseZeroAgent
from training.loop import TrainState, resident_train_step
from optimization.hardware_compat import setup_jax_platform
import time

def main():
    print("Initializing RecurseZero...")
    setup_jax_platform()
    
    # Apply Metal Patch for Pgx (Simulate GPU-Safe Hashing)
    from env.pgx_patch import apply_patch
    apply_patch()
    
    # 1. Initialize Environment
    # 8GB VRAM constraint -> batch size careful tuning
    BATCH_SIZE = 2048 
    env = RecurseEnv(batch_size=BATCH_SIZE)
    key = jax.random.PRNGKey(42)
    key, env_key = jax.random.split(key)
    
    # Initialize env state on GPU
    env_state = env.init(env_key)
    print(f"Environment initialized with Batch Size: {BATCH_SIZE}, Actions: {env.num_actions}")
    
    # 2. Initialize Model
    # We pass the environment's action space size to the agent
    agent = RecurseZeroAgent(num_actions=env.num_actions)
    
    # Get dummy input for init
    dummy_obs = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    # 3. Optimize Setup
    # "AdamW optimizer"
    optimizer = optax.adamw(learning_rate=3e-4) # Standard start
    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=optimizer
    )
    print("Model and Optimizer initialized.")
    
    # 4. Training Loop
    # We run the loop using the JIT-compiled resident step function.
    NUM_STEPS = 50 # Verification of Perfection
    
    # JIT the step function with the static callables (agent.apply, env.step)
    # This compiles the ENTIRE loop body (Env + Agent) into one kernel.
    step_fn = resident_train_step
    
    print(f"Starting Resident Training Loop for {NUM_STEPS} steps...")
    start_time = time.time()
    
    for i in range(NUM_STEPS):
        key, step_key = jax.random.split(key)
        
        train_state, env_state, metrics = step_fn(
            train_state, 
            env_state, 
            step_key, 
            agent.apply, 
            env.step
        )
        
        if i % 100 == 0:
             # Blokcing print for monitoring
             loss_val = metrics['total_loss']
             reward_val = metrics['mean_reward']
             print(f"Step {i}: Loss={loss_val:.4f}, Reward={reward_val:.4f}")
            
    end_time = time.time()
    print(f"Training finished. {NUM_STEPS} steps in {end_time - start_time:.2f}s")
    print(f"Steps per second: {NUM_STEPS / (end_time - start_time):.2f}")
    
if __name__ == "__main__":
    main()
