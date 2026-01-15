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
    IMPROVED Resident Training Step.
    
    Fixes:
    - Stronger exploration to avoid draw convergence
    - Win bonus to encourage decisive play
    - Correct loss counting (losses = wins in self-play)
    """
    batch_size = env_state.observation.shape[0]
    
    obs = env_state.observation
    current_player = env_state.current_player
    
    # Forward pass
    policy_logits, values, reward_pred = agent_apply_fn(state.params, obs)
    
    # Mask illegal moves
    legal_mask = env_state.legal_action_mask
    masked_logits = jnp.where(legal_mask, policy_logits, -1e9)
    
    # STRONGER EXPLORATION: Higher temperature early in training
    temperature = 1.5  # Increased from 1.0 for more exploration
    scaled_logits = masked_logits / temperature
    
    key, action_key = jax.random.split(key)
    actions = jax.random.categorical(action_key, scaled_logits)
    
    # Step Environment
    next_env_state = env_step_fn(env_state, actions)
    
    # Extract rewards
    rewards_all = next_env_state.rewards
    rewards = rewards_all[jnp.arange(batch_size), current_player]
    
    terminated = next_env_state.terminated
    
    # WIN BONUS: Give extra reward for decisive games to discourage draws
    is_win = (terminated) & (rewards > 0.5)
    is_draw = (terminated) & (jnp.abs(rewards) < 0.5)
    
    # Penalize draws slightly to encourage aggressive play
    rewards = jnp.where(is_draw, -0.1, rewards)  # Small penalty for draws
    rewards = jnp.where(is_win, rewards * 1.2, rewards)  # Bonus for wins
    
    # Compute TD targets
    next_obs = next_env_state.observation
    _, next_values, _ = agent_apply_fn(state.params, next_obs)
    next_vals = jnp.squeeze(next_values, -1)
    next_vals = jnp.where(terminated, 0.0, next_vals)
    
    vals = jnp.squeeze(values, -1)
    
    gamma = 0.99
    target_values = rewards + gamma * next_vals
    advantages = target_values - vals
    advantages = jnp.clip(advantages, -1.0, 1.0)
    
    # Loss function with STRONGER entropy bonus
    def loss_fn(params):
        p_logits, v_pred, r_pred = agent_apply_fn(params, obs)
        v_pred = jnp.squeeze(v_pred, -1)
        r_pred = jnp.squeeze(r_pred, -1)
        
        masked_p = jnp.where(legal_mask, p_logits, -1e9)
        
        probs = jax.nn.softmax(masked_p)
        log_probs = jax.nn.log_softmax(masked_p)
        
        one_hot = jax.nn.one_hot(actions, p_logits.shape[-1])
        selected_log_probs = jnp.sum(log_probs * one_hot, axis=-1)
        pg_loss = -jnp.mean(selected_log_probs * advantages)
        
        # STRONGER entropy bonus (was 0.01)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        entropy_bonus = -0.05 * jnp.mean(entropy)  # 5x stronger!
        
        # Value loss
        value_loss = jnp.mean(jnp.square(v_pred - target_values))
        
        # Reward loss
        reward_loss = jnp.mean(jnp.where(terminated, jnp.square(r_pred - rewards), 0.0))
        
        total = pg_loss + entropy_bonus + 0.5 * value_loss + 0.1 * reward_loss
        return total, (pg_loss, value_loss, entropy)

    grads, (pg_loss, v_loss, entropy) = jax.grad(loss_fn, has_aux=True)(state.params)
    
    # Gradient clipping
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # Optimizer step
    new_state = state.apply_gradients(grads=grads)
    
    # Auto-reset terminated games
    if env_init_fn is not None:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        fresh_states = env_init_fn(reset_keys)
        
        def select(fresh, current):
            mask = terminated.reshape((-1,) + (1,) * (len(current.shape) - 1))
            return jnp.where(mask, fresh, current)
        
        next_env_state = jax.tree.map(select, fresh_states, next_env_state)
    
    # FIXED METRICS: In self-play, losses = wins (symmetric)
    games_done = jnp.sum(terminated)
    
    wins = jnp.sum((terminated) & (rewards > 0.5))
    draws = jnp.sum((terminated) & (jnp.abs(rewards) <= 0.5) & (rewards > -0.5))
    decisive = wins  # In self-play, each decisive game has 1 win and 1 loss
    losses = decisive  # FIXED: losses equals wins in self-play!
    
    metrics = {
        'total_loss': pg_loss + 0.5 * v_loss,
        'pg_loss': pg_loss,
        'value_loss': v_loss,
        'mean_reward': jnp.mean(jnp.where(terminated, rewards, 0.0)),
        'policy_entropy': jnp.mean(entropy),
        'mean_value': jnp.mean(values),
        'games_finished': games_done,
        'wins': wins,
        'losses': losses,  # Now correctly equals wins
        'draws': draws,
        'decisive_games': decisive,
    }
    
    return new_state, next_env_state, metrics
