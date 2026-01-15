import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

from algorithm.muesli import muesli_policy_gradient_loss, compute_muesli_targets
from algorithm.pve import pve_loss

class TrainState(train_state.TrainState):
    # We can add extra fields if needed, e.g. target network params
    pass

@partial(jax.jit, static_argnames=('agent_apply_fn', 'env_step_fn'))
def resident_train_step(
    state: TrainState,
    env_state,
    key: jax.Array,
    agent_apply_fn,
    env_step_fn
):
    """
    Executes one step of Resident Training (Env Step + Model Update).
    Compiled into a single XLA kernel.
    """
    # 1. Environment Interaction
    # We use the CURRENT model to select actions (Muesli: Search-Free)
    obs = env_state.observation
    
    # Run Inference (No Gradients)
    policy_logits, values, _ = agent_apply_fn(state.params, obs)
    
    # Sample Actions (Argmax for stability/perfection)
    actions = jnp.argmax(policy_logits, axis=-1)
    
    # Step Environment (Pgx)
    next_env_state = env_step_fn(env_state, actions)
    next_obs = next_env_state.observation
    
    # Correct Rewards Extraction
    player_indices = env_state.current_player[:, None] # (B, 1)
    rewards = jnp.take_along_axis(next_env_state.rewards, player_indices, axis=1) # (B, 1)
    rewards = rewards.reshape(-1, 1) # (B, 1)
    
    # 2. Retrace / Targets
    # Get Value of Next State
    _, next_values, _ = agent_apply_fn(state.params, next_obs)
    
    # Compute Targets
    vals = jnp.squeeze(values, -1)
    next_vals = jnp.squeeze(next_values, -1)
    rewards_flat = jnp.squeeze(rewards, -1)
    
    advantages, target_values = compute_muesli_targets(vals, rewards_flat, next_vals)
    
    # 3. Model Update (Loss Calculation)
    def loss_fn(params):
        # Forward pass on CURRENT observation
        p_logits, v_pred, r_pred = agent_apply_fn(params, obs)
        v_pred = jnp.squeeze(v_pred, -1)
        r_pred = jnp.squeeze(r_pred, -1)
        
        # Policy Loss
        pg_loss = muesli_policy_gradient_loss(p_logits, actions, advantages)
        
        # PVE Loss
        loss_pve = pve_loss(v_pred, target_values, r_pred, rewards_flat)
        
        total_loss = pg_loss + loss_pve
        return total_loss, (pg_loss, loss_pve)

    grads, (pg_loss, pve_loss_val) = jax.grad(loss_fn, has_aux=True)(state.params)
    
    # 4. Optimizer Step
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'total_loss': pg_loss + pve_loss_val,
        'pg_loss': pg_loss,
        'pve_loss': pve_loss_val,
        'mean_reward': jnp.mean(rewards_flat)
    }
    
    return new_state, next_env_state, metrics
