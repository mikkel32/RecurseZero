import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

from algorithm.muesli import muesli_policy_gradient_loss, compute_muesli_targets
from algorithm.pve import pve_loss

class TrainState(train_state.TrainState):
    # Extended state to track episodic metrics
    pass

@partial(jax.jit, static_argnames=('agent_apply_fn', 'env_step_fn', 'env_init_fn'))
def resident_train_step(
    state: TrainState,
    env_state,
    key: jax.Array,
    agent_apply_fn,
    env_step_fn,
    env_init_fn=None  # For resetting terminated games
):
    """
    Executes one step of Resident Training (Env Step + Model Update).
    Compiled into a single XLA kernel.
    
    Key improvements:
    - Handles game termination and auto-reset
    - Proper reward extraction at game end
    - Tracks episodic returns
    """
    batch_size = env_state.observation.shape[0]
    
    # 1. Environment Interaction
    obs = env_state.observation
    
    # Run Inference (No Gradients) - Get action distribution
    policy_logits, values, _ = agent_apply_fn(state.params, obs)
    
    # Mask illegal moves (set logits to -inf)
    legal_mask = env_state.legal_action_mask
    masked_logits = jnp.where(legal_mask, policy_logits, -1e9)
    
    # Sample from policy (exploration) instead of argmax
    key, action_key = jax.random.split(key)
    actions = jax.random.categorical(action_key, masked_logits)
    
    # Step Environment (Pgx)
    next_env_state = env_step_fn(env_state, actions)
    
    # 2. Proper Reward Extraction
    # Rewards in Pgx are [player0_reward, player1_reward] per position
    # We need the reward for the current player who just moved
    player_indices = env_state.current_player  # (B,)
    # rewards shape is (B, 2) - one for each player
    rewards = next_env_state.rewards[jnp.arange(batch_size), player_indices]  # (B,)
    
    # Terminal mask - games that just ended
    terminated = next_env_state.terminated  # (B,) boolean
    
    # 3. Auto-Reset terminated games
    if env_init_fn is not None:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        fresh_states = env_init_fn(reset_keys)
        
        # Selectively replace terminated games with fresh states
        # Use jax.tree_map to handle the state structure
        def select_state(fresh, current, mask):
            # mask is True where we want fresh (terminated games)
            return jnp.where(
                mask.reshape((-1,) + (1,) * (len(current.shape) - 1)),
                fresh,
                current
            )
        
        next_env_state = jax.tree.map(
            lambda f, c: select_state(f, c, terminated),
            fresh_states,
            next_env_state
        )
    
    next_obs = next_env_state.observation
    
    # 4. Compute Targets with terminal handling
    _, next_values, _ = agent_apply_fn(state.params, next_obs)
    
    vals = jnp.squeeze(values, -1)
    next_vals = jnp.squeeze(next_values, -1)
    
    # Zero out next_vals for terminated games (no future value)
    next_vals = jnp.where(terminated, 0.0, next_vals)
    
    advantages, target_values = compute_muesli_targets(vals, rewards, next_vals)
    
    # 5. Model Update (Loss Calculation)
    def loss_fn(params):
        p_logits, v_pred, r_pred = agent_apply_fn(params, obs)
        v_pred = jnp.squeeze(v_pred, -1)
        r_pred = jnp.squeeze(r_pred, -1)
        
        # Policy Loss with legal move masking
        masked_p_logits = jnp.where(legal_mask, p_logits, -1e9)
        pg_loss = muesli_policy_gradient_loss(masked_p_logits, actions, advantages)
        
        # PVE Loss
        loss_pve = pve_loss(v_pred, target_values, r_pred, rewards)
        
        total_loss = pg_loss + loss_pve
        return total_loss, (pg_loss, loss_pve)

    grads, (pg_loss, pve_loss_val) = jax.grad(loss_fn, has_aux=True)(state.params)
    
    # Gradient clipping for stability
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # 6. Optimizer Step
    new_state = state.apply_gradients(grads=grads)
    
    # 7. Calculate Metrics
    # Policy Entropy (over legal moves only)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    log_probs = jnp.log(probs + 1e-8)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    mean_entropy = jnp.mean(entropy)
    
    # Win rate from terminated games
    terminal_rewards = jnp.where(terminated, rewards, 0.0)
    wins = jnp.sum(terminal_rewards > 0)
    losses = jnp.sum(terminal_rewards < 0)
    draws = jnp.sum((terminated) & (terminal_rewards == 0))
    games_finished = jnp.sum(terminated)
    
    metrics = {
        'total_loss': pg_loss + pve_loss_val,
        'pg_loss': pg_loss,
        'pve_loss': pve_loss_val,
        'mean_reward': jnp.mean(rewards),
        'policy_entropy': mean_entropy,
        'mean_value': jnp.mean(values),
        'games_finished': games_finished,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'terminal_reward': jnp.sum(jnp.abs(terminal_rewards)),  # Total reward from finished games
    }
    
    return new_state, next_env_state, metrics
