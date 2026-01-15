import jax
import jax.numpy as jnp
import flax.linen as nn

def pve_loss(pred_value, target_value, pred_reward=None, target_reward=None):
    """
    Computes the Proper Value Equivalence loss.
    
    The model should learn a state representation z* such that linear (or simple) probes
    can predict the value and reward.
    
    Args:
        pred_value: Predicted value logits or scalar (Batch, 1) or (Batch, buckets)
        target_value: Ground truth value (Batch, 1)
        pred_reward: Predicted reward (optional for chess if only terminal)
        target_reward: Ground truth reward
        
    Returns:
        Scalar loss
    """
    # Muesli / PVE often uses categorical cross-entropy for values (MuZero style) 
    # or simple MSE/Huber for scalar.
    # Text section 5.2: "Value Head outputs a scalar v in [-1, 1]"
    # This implies Tanh activation and MSE or similar loss.
    
    value_loss = jnp.mean(jnp.square(pred_value - target_value))
    
    loss = value_loss
    
    if pred_reward is not None and target_reward is not None:
        reward_loss = jnp.mean(jnp.square(pred_reward - target_reward))
        loss += reward_loss
        
    return loss

class ValueHead(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x) # Output in [-1, 1]
        return x

class RewardHead(nn.Module):
    # For chess, reward is 0 usually until end?
    # Or intermediate rewards?
    # Muesli uses rewards. We provide head just in case.
    hidden_dim: int
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        # Reward can be unbounded? Or [-1, 1]?
        # Chess result is -1, 0, 1.
        return x
