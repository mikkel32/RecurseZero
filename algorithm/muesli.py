"""
Muesli Policy Optimization for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec section 3.1-3.2:
- Search-free policy inference
- Clipped MPO (CMPO) advantage estimation
- Policy gradient with entropy bonus
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class PolicyHead(nn.Module):
    """
    Policy head for Muesli algorithm.
    
    Outputs raw logits over all possible moves.
    Uses same dtype as input (BF16 or FP32).
    """
    hidden_dim: int
    num_actions: int  # 4672 for Pgx chess
    
    @nn.compact
    def __call__(self, z):
        # Hidden layer
        x = nn.Dense(self.hidden_dim, name='fc1')(z)
        x = nn.gelu(x)
        
        # Output logits (always FP32 for stable softmax)
        logits = nn.Dense(self.num_actions, name='fc2')(x)
        logits = logits.astype(jnp.float32)
        
        return logits


def muesli_policy_gradient_loss(policy_logits, actions, advantages):
    """
    Computes the Policy Gradient loss with CMPO clipping.
    
    L = -E[log π(a|s) · A(s,a)]
    
    Args:
        policy_logits: (Batch, NumActions)
        actions: (Batch,) - Action indices
        advantages: (Batch,) - Clipped advantages
        
    Returns:
        loss: Scalar policy gradient loss
    """
    one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
    log_probs = nn.log_softmax(policy_logits)
    
    selected_log_probs = jnp.sum(log_probs * one_hot, axis=-1)
    loss = -jnp.mean(selected_log_probs * advantages)
    
    return loss


def compute_muesli_targets(values, rewards, next_values, gamma=0.99):
    """
    Computes Retrace-style TD targets and advantages.
    
    Args:
        values: Current value predictions
        rewards: Observed rewards
        next_values: Next state value predictions
        gamma: Discount factor
        
    Returns:
        advantages: Clipped advantage estimates
        target_values: TD target values
    """
    target_values = rewards + gamma * next_values
    advantages = target_values - values
    
    # CMPO clipping for stability (spec 3.2)
    advantages = jnp.clip(advantages, -1.0, 1.0)
    
    return advantages, target_values


def policy_entropy(logits, legal_mask=None):
    """
    Compute entropy of policy distribution.
    
    H(π) = -Σ π(a) log π(a)
    
    Args:
        logits: Policy logits (B, A)
        legal_mask: Optional mask for legal actions
        
    Returns:
        entropy: Per-sample entropy (B,)
    """
    if legal_mask is not None:
        logits = jnp.where(legal_mask, logits, -1e9)
    
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    return -jnp.sum(probs * log_probs, axis=-1)
