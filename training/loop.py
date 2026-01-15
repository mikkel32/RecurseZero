import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

from algorithm.muesli import muesli_policy_gradient_loss, compute_muesli_targets
from algorithm.pve import pve_loss

class TrainState(train_state.TrainState):
    pass

@partial(jax.jit, static_argnames=('agent_apply_fn', 'env_step_fn', 'env_init_fn'))
def resident_train_step(
    state: TrainState,
    env_state,
    key: jax.Array,
    agent_apply_fn,
    env_step_fn,
    env_init_fn=None
):
    """
    SPEED OPTIMIZED Resident Training Step.
    
    Changes:
    - Single forward pass (no separate next_values call)
    - Simplified game reset logic
    - Proper reward extraction
    """
    batch_size = env_state.observation.shape[0]
    
    # 1. Get current state info
    obs = env_state.observation
    current_player = env_state.current_player
    
    # Run Inference
    policy_logits, values, reward_pred = agent_apply_fn(state.params, obs)
    
    # Mask illegal moves and sample
    legal_mask = env_state.legal_action_mask
    masked_logits = jnp.where(legal_mask, policy_logits, -1e9)
    
    key, action_key = jax.random.split(key)
    actions = jax.random.categorical(action_key, masked_logits)
    
    # Step Environment
    next_env_state = env_step_fn(env_state, actions)
    
    # 2. Extract rewards CORRECTLY
    # In Pgx chess: rewards is (B, 2) where [player0_reward, player1_reward]
    # We want the reward for the player who just moved (current_player BEFORE step)
    rewards_all = next_env_state.rewards  # (B, 2)
    rewards = rewards_all[jnp.arange(batch_size), current_player]  # (B,)
    
    # Terminal detection
    terminated = next_env_state.terminated
    
    # 3. Compute TD target (bootstrap from next state value)
    # Only call forward pass on next state for non-terminated games
    next_obs = next_env_state.observation
    _, next_values, _ = agent_apply_fn(state.params, next_obs)
    next_vals = jnp.squeeze(next_values, -1)
    
    # Zero out next values for terminated games
    next_vals = jnp.where(terminated, 0.0, next_vals)
    
    vals = jnp.squeeze(values, -1)
    
    # Compute targets
    gamma = 0.99
    target_values = rewards + gamma * next_vals
    advantages = target_values - vals
    advantages = jnp.clip(advantages, -1.0, 1.0)
    
    # 4. Loss function
    def loss_fn(params):
        p_logits, v_pred, r_pred = agent_apply_fn(params, obs)
        v_pred = jnp.squeeze(v_pred, -1)
        r_pred = jnp.squeeze(r_pred, -1)
        
        # Mask illegal for policy loss
        masked_p = jnp.where(legal_mask, p_logits, -1e9)
        
        # Policy gradient loss
        one_hot = jax.nn.one_hot(actions, p_logits.shape[-1])
        log_probs = jax.nn.log_softmax(masked_p)
        selected_log_probs = jnp.sum(log_probs * one_hot, axis=-1)
        pg_loss = -jnp.mean(selected_log_probs * advantages)
        
        # Value loss  
        value_loss = jnp.mean(jnp.square(v_pred - target_values))
        
        # Reward prediction loss (only for terminal states)
        reward_loss = jnp.mean(jnp.where(terminated, jnp.square(r_pred - rewards), 0.0))
        
        total = pg_loss + 0.5 * value_loss + 0.1 * reward_loss
        return total, (pg_loss, value_loss, reward_loss)

    grads, (pg_loss, v_loss, r_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
    
    # Gradient clipping
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # 5. Optimizer step
    new_state = state.apply_gradients(grads=grads)
    
    # 6. Auto-reset terminated games (simplified)
    if env_init_fn is not None:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        fresh_states = env_init_fn(reset_keys)
        
        # Broadcast terminated mask for each field
        def select(fresh, current):
            mask = terminated.reshape((-1,) + (1,) * (len(current.shape) - 1))
            return jnp.where(mask, fresh, current)
        
        next_env_state = jax.tree.map(select, fresh_states, next_env_state)
    
    # 7. Metrics
    probs = jax.nn.softmax(masked_logits, axis=-1)
    log_probs_m = jnp.log(probs + 1e-8)
    entropy = -jnp.sum(probs * log_probs_m, axis=-1)
    
    # Count outcomes
    games_done = jnp.sum(terminated)
    wins = jnp.sum((terminated) & (rewards > 0.5))
    losses = jnp.sum((terminated) & (rewards < -0.5))
    draws = jnp.sum((terminated) & (jnp.abs(rewards) < 0.5))
    
    metrics = {
        'total_loss': pg_loss + 0.5 * v_loss + 0.1 * r_loss,
        'pg_loss': pg_loss,
        'value_loss': v_loss,
        'mean_reward': jnp.mean(rewards),
        'terminal_rewards': jnp.sum(jnp.where(terminated, rewards, 0.0)),
        'policy_entropy': jnp.mean(entropy),
        'mean_value': jnp.mean(values),
        'games_finished': games_done,
        'wins': wins,
        'losses': losses,
        'draws': draws,
    }
    
    return new_state, next_env_state, metrics
