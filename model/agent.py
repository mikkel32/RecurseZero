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
    
    Structure:
    Input (Board) -> Embedding -> DEQ -> Abstract State z*
    z* -> Policy Head -> Logits
    z* -> Value Head -> V(s)
    z* -> Reward Head -> R(s)
    """
    hidden_dim: int = 256
    heads: int = 4
    mlp_dim: int = 1024
    num_actions: int = 4672 # Default Chess Action Space size, but should be set by instance
    
    @nn.compact
    def __call__(self, x):
        # x: Input observation using Pgx encoding (Batch, 8, 8, C)
        
        # 1. Embedding
        # Project 8x8xC to 8x8xDim
        x_embed = nn.Dense(self.hidden_dim)(x)
        # Flatten for Transformer: (Batch, 64, Dim)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core
        # Uses Universal Transformer Block
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim
            }
        )
        
        z_star = deq(x_flat)
        
        # 3. Heads
        # We might pool z* or use the whole state?
        # Usually for Policy/Value we pool or use specialized tokens?
        # Standard: Mean pooling or [CLS] token equivalent.
        # Or simple flatten? Flatten is huge (64*256).
        # We'll use Mean Pooling over the board squares.
        
        z_pooled = jnp.mean(z_star, axis=1) # (Batch, Dim)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions)(z_pooled)
        value = ValueHead(self.hidden_dim)(z_pooled)
        reward = RewardHead(self.hidden_dim)(z_pooled)
        
        return policy_logits, value, reward
