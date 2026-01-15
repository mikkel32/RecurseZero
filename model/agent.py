import jax
import jax.numpy as jnp
import flax.linen as nn
from .deq import DeepEquilibriumModel
from .universal_transformer import UniversalTransformerBlock
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead

class RecurseZeroAgent(nn.Module):
    """
    The full RecurseZero Agent.
    
    SPEED OPTIMIZED:
    - Reduced hidden_dim: 256 -> 128
    - Reduced mlp_dim: 1024 -> 512
    - DEQ iterations: 6 -> 3
    
    Structure:
    Input (Board) -> Embedding -> DEQ -> Abstract State z*
    z* -> Policy Head -> Logits
    z* -> Value Head -> V(s)
    z* -> Reward Head -> R(s)
    """
    # SPEED OPTIMIZED DEFAULTS
    hidden_dim: int = 128  # Reduced from 256
    heads: int = 4
    mlp_dim: int = 512     # Reduced from 1024
    num_actions: int = 4672
    deq_iters: int = 3     # Reduced from 6
    
    @nn.compact
    def __call__(self, x):
        # x: Input observation using Pgx encoding (Batch, 8, 8, C)
        
        # 1. Embedding
        x_embed = nn.Dense(self.hidden_dim)(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core with reduced iterations
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim
            },
            max_iter=self.deq_iters,  # Fast mode
            m=2,  # Smaller Anderson history
            beta=0.9  # Higher damping for faster convergence
        )
        
        z_star = deq(x_flat)
        
        # 3. Heads with mean pooling
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions)(z_pooled)
        value = ValueHead(self.hidden_dim)(z_pooled)
        reward = RewardHead(self.hidden_dim)(z_pooled)
        
        return policy_logits, value, reward
