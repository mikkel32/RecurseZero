"""
RecurseZero Agent with BFloat16 Mixed Precision.

Per GPU ONLY Chess RL Agent.txt spec:
- DEQ core with Anderson Acceleration
- GTrXL stabilization
- Chess-aware positional encodings
- Muesli policy head (search-free)
- PVE value/reward heads
- BFloat16 for 2x memory savings and faster compute
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .deq import DeepEquilibriumModel
from .universal_transformer import UniversalTransformerBlock
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead

# Get compute dtype
try:
    from optimization.mixed_precision import get_dtype, is_bf16_enabled
except ImportError:
    def get_dtype():
        return jnp.float32
    def is_bf16_enabled():
        return False


class RecurseZeroAgent(nn.Module):
    """
    Full RecurseZero Agent with BFloat16 mixed precision.
    
    Structure:
    Input (Board) -> Embedding -> DEQ -> Abstract State z*
    z* -> Policy Head -> Logits
    z* -> Value Head -> V(s)
    z* -> Reward Head -> R(s)
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
        # Get compute dtype (BF16 or FP32)
        compute_dtype = get_dtype()
        
        # Cast input to compute dtype
        x = x.astype(compute_dtype)
        
        # 1. Input Embedding (in BF16)
        x_embed = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim,
                'dtype': compute_dtype,  # Pass dtype to block
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        
        # 3. Output Heads (cast back to FP32 for stability)
        z_pooled = jnp.mean(z_star, axis=1).astype(jnp.float32)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


class RecurseZeroAgentFast(nn.Module):
    """
    Speed-optimized version with BFloat16 for L4 GPU.
    
    Reduced parameters + BF16 = maximum throughput.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_actions: int = 4672
    deq_iters: int = 4
    deq_tol: float = 1e-2
    deq_beta: float = 0.9
    deq_m: int = 2
    
    @nn.compact
    def __call__(self, x):
        # Get compute dtype (BF16 or FP32)
        compute_dtype = get_dtype()
        
        # Cast input
        x = x.astype(compute_dtype)
        
        # 1. Embedding
        x_embed = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core with BF16
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim,
                'dtype': compute_dtype,
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        
        # 3. Output Heads (FP32)
        z_pooled = jnp.mean(z_star, axis=1).astype(jnp.float32)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward
