"""
Proper Value Equivalence (PVE) for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec section 3.3:
- Model learns abstract state representations for planning
- Predicts value V(s) and reward R(s)

Per spec section 5.2 (Confidence and Win-Rate Estimation):
- Value v ∈ [-1, 1] maps to P(win) = (v+1)/2
- Entropy metric tracks calculation confidence
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


def pve_loss(
    pred_value: jnp.ndarray, 
    target_value: jnp.ndarray, 
    pred_reward: jnp.ndarray = None, 
    target_reward: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Computes the Proper Value Equivalence loss.
    
    The model learns a state representation z* such that simple probes
    can predict the value and reward.
    
    Args:
        pred_value: Predicted value (Batch, 1) in [-1, 1]
        target_value: Ground truth value (Batch, 1)
        pred_reward: Predicted reward (optional)
        target_reward: Ground truth reward
        
    Returns:
        loss: Scalar PVE loss
    """
    value_loss = jnp.mean(jnp.square(pred_value - target_value))
    
    loss = value_loss
    
    if pred_reward is not None and target_reward is not None:
        reward_loss = jnp.mean(jnp.square(pred_reward - target_reward))
        loss = loss + reward_loss
        
    return loss


def value_to_win_probability(value: jnp.ndarray) -> jnp.ndarray:
    """
    Convert value prediction to win probability.
    
    Per spec section 5.2:
    "Value v ∈ [-1, 1] maps directly to P(win) = (v+1)/2"
    
    Args:
        value: Value prediction in [-1, 1]
        
    Returns:
        win_prob: Probability of winning in [0, 1]
    """
    return (value + 1.0) / 2.0


def win_probability_to_centipawn(win_prob: jnp.ndarray) -> jnp.ndarray:
    """
    Convert win probability to centipawn evaluation.
    
    Uses logistic function inverse: cp = -log((1-p)/p) * 100
    
    Args:
        win_prob: Win probability in (0, 1)
        
    Returns:
        centipawn: Evaluation in centipawns (capped at ±1000)
    """
    # Clamp to avoid log(0)
    win_prob = jnp.clip(win_prob, 0.001, 0.999)
    
    # Logistic inverse
    cp = -jnp.log((1.0 - win_prob) / win_prob) * 100.0
    
    # Cap at reasonable bounds
    return jnp.clip(cp, -1000.0, 1000.0)


class ValueHead(nn.Module):
    """
    Value prediction head for PVE.
    
    Outputs scalar value v ∈ [-1, 1] representing expected game outcome.
    - v = +1: White wins
    - v = 0: Draw
    - v = -1: Black wins (from white's perspective)
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name='fc1')(z)
        x = nn.gelu(x)
        x = nn.Dense(1, name='fc2')(x)
        x = nn.tanh(x)  # Output in [-1, 1]
        return x
    
    @staticmethod
    def get_win_probability(value: jnp.ndarray) -> jnp.ndarray:
        """Convert value to win probability."""
        return value_to_win_probability(value)
    
    @staticmethod
    def get_centipawn_eval(value: jnp.ndarray) -> jnp.ndarray:
        """Convert value to centipawn evaluation."""
        win_prob = value_to_win_probability(value)
        return win_probability_to_centipawn(win_prob)


class RewardHead(nn.Module):
    """
    Reward prediction head for PVE.
    
    Predicts immediate reward for the transition.
    In chess: 0 for most moves, ±1 at game end.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name='fc1')(z)
        x = nn.gelu(x)
        x = nn.Dense(1, name='fc2')(x)
        # No activation - reward can be any value
        return x


class ConfidenceMetrics:
    """
    Compute confidence metrics for interpretability.
    
    Per spec section 5.2:
    - Policy entropy for uncertainty tracking
    - DEQ residual for positional complexity
    """
    
    @staticmethod
    def policy_entropy(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Compute entropy of policy distribution.
        
        H(π) = -Σ π(a) log π(a)
        
        High entropy = uncertain/complex position
        Low entropy = confident in best move
        
        Args:
            logits: Policy logits (B, A)
            legal_mask: Legal action mask (B, A)
            
        Returns:
            entropy: Per-sample entropy (B,)
        """
        masked_logits = jnp.where(legal_mask, logits, -1e9)
        probs = jax.nn.softmax(masked_logits, axis=-1)
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        
        # Entropy only over legal moves
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        
        return entropy
    
    @staticmethod
    def confidence_score(entropy: jnp.ndarray, max_entropy: float = 8.0) -> jnp.ndarray:
        """
        Convert entropy to confidence score in [0, 1].
        
        Args:
            entropy: Policy entropy values
            max_entropy: Maximum expected entropy (log of avg legal moves)
            
        Returns:
            confidence: 1 = very confident, 0 = uncertain
        """
        return jnp.clip(1.0 - entropy / max_entropy, 0.0, 1.0)
    
    @staticmethod
    def format_evaluation(
        value: float,
        entropy: float,
        residual: float = None
    ) -> dict:
        """
        Format evaluation metrics for display.
        
        Args:
            value: Value prediction in [-1, 1]
            entropy: Policy entropy
            residual: DEQ convergence residual (optional)
            
        Returns:
            metrics: Dictionary with formatted values
        """
        win_prob = float((value + 1.0) / 2.0)
        confidence = float(max(0, 1.0 - entropy / 8.0))
        
        # Centipawn conversion
        win_prob_clipped = max(0.001, min(0.999, win_prob))
        cp = -100.0 * float(jnp.log((1.0 - win_prob_clipped) / win_prob_clipped))
        cp = max(-1000, min(1000, cp))
        
        metrics = {
            'win_probability': f"{win_prob:.1%}",
            'centipawn': f"{cp:+.0f}",
            'confidence': f"{confidence:.1%}",
            'raw_value': f"{value:.3f}",
        }
        
        if residual is not None:
            complexity = "Easy" if residual < 0.01 else "Medium" if residual < 0.1 else "Hard"
            metrics['complexity'] = complexity
            metrics['residual'] = f"{residual:.4f}"
            
        return metrics
