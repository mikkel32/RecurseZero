"""
RecurseZero Agent - Fast and Stable.

Per GPU ONLY Chess RL Agent.txt spec:
- DEQ core with Anderson Acceleration
- GTrXL stabilization
- Chess-aware positional encodings
- Muesli policy head (search-free)
- PVE value/reward heads

Optimized for L4 GPU with FP32 training.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .deq import DeepEquilibriumModel
from .universal_transformer import UniversalTransformerBlock
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead


class RecurseZeroAgent(nn.Module):
    """
    Full RecurseZero Agent (spec-compliant).
    
    Uses larger dimensions for maximum performance.
    """
    hidden_dim: int = 256
    heads: int = 8
    mlp_dim: int = 1024
    num_actions: int = 4672
    deq_iters: int = 8
    deq_tol: float = 1e-3
    deq_beta: float = 0.9
    deq_m: int = 3
    
    @nn.compact
    def __call__(self, x):
        # 1. Input Embedding
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim,
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        
        # 3. Output Heads
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


class RecurseZeroAgentFast(nn.Module):
    """
    Speed-optimized agent for L4 GPU.
    
    Aggressively reduced parameters for fast training.
    Target: 3+ steps/second on L4.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_actions: int = 4672
    deq_iters: int = 3       # Reduced from 4
    deq_tol: float = 5e-2    # Looser tolerance
    deq_beta: float = 0.9
    deq_m: int = 2
    
    @nn.compact
    def __call__(self, x):
        # 1. Embedding
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim,
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        
        # 3. Output Heads
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward
