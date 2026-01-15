import jax
import jax.numpy as jnp
import flax.linen as nn

class PolicyHead(nn.Module):
    hidden_dim: int
    num_actions: int # Chess typically ~1858 to 4672? Pgx defines action space.
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.gelu(x)
        # Final projection to action logits
        logits = nn.Dense(self.num_actions)(x)
        return logits



def muesli_policy_gradient_loss(policy_logits, actions, advantages):
    """
    Computes the Policy Gradient loss compatible with Muesli's advantage.
    
    L = - sum ( log_pi(a_t) * A_t )
    
    Args:
        policy_logits: (Batch, NumActions)
        actions: (Batch,) - Indices of actions taken
        advantages: (Batch,) - Calculated advantages (Retrace)
    """
    one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
    log_probs = nn.log_softmax(policy_logits)
    
    # Select log prob of the action taken
    selected_log_probs = jnp.sum(log_probs * one_hot, axis=-1)
    
    # Loss is negative expected reward (advantage)
    loss = -jnp.mean(selected_log_probs * advantages)
    return loss

def compute_muesli_targets(values, rewards, next_values, gamma=0.99):
    """
    Computes Retrace-like advantages.
    """
    target_values = rewards + gamma * next_values
    advantages = target_values - values
    
    # Clipping for stability (CMPO)
    advantages = jnp.clip(advantages, -1.0, 1.0)
    
    return advantages, target_values
