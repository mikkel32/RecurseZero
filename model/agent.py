"""
RecurseZero Agent - Multiple Variants for Testing.

Includes:
- RecurseZeroAgent: Full spec-compliant version
- RecurseZeroAgentFast: DEQ with minimal iterations
- RecurseZeroAgentSimple: NO DEQ, just stacked layers (fastest)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE TRANSFORMER (NO DEQ) - For speed comparison
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTransformerBlock(nn.Module):
    """Standard transformer block without DEQ overhead."""
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    @nn.compact
    def __call__(self, x):
        # 1. Multi-head attention
        h = nn.LayerNorm()(x)
        
        # Split into Q, K, V
        head_dim = self.hidden_dim // self.heads
        qkv = nn.Dense(3 * self.hidden_dim)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for attention
        B, L, _ = q.shape
        q = q.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.float32(head_dim))
        attn = nn.softmax(jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale, axis=-1)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_dim)
        out = nn.Dense(self.hidden_dim)(out)
        
        x = x + out
        
        # 2. MLP
        h = nn.LayerNorm()(x)
        mlp = nn.Dense(self.mlp_dim)(h)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.hidden_dim)(mlp)
        x = x + mlp
        
        return x


class RecurseZeroAgentSimple(nn.Module):
    """
    Simple agent WITHOUT DEQ - for speed baseline.
    
    Uses standard stacked transformer layers instead of fixed-point iteration.
    Should be MUCH faster than DEQ version.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_layers: int = 3     # Stacked layers instead of DEQ iterations
    num_actions: int = 4672
    
    @nn.compact
    def __call__(self, x):
        # 1. Embedding
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. Stacked transformer blocks (NO DEQ)
        h = x_flat
        for i in range(self.num_layers):
            h = SimpleTransformerBlock(
                hidden_dim=self.hidden_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                name=f'block_{i}'
            )(h)
        
        # 3. Output heads
        z_pooled = jnp.mean(h, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


# ═══════════════════════════════════════════════════════════════════════════════
# DEQ VERSIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Import DEQ only when needed
def _get_deq_agent():
    from .deq import DeepEquilibriumModel
    from .universal_transformer import UniversalTransformerBlock
    return DeepEquilibriumModel, UniversalTransformerBlock


class RecurseZeroAgent(nn.Module):
    """Full spec-compliant DEQ agent."""
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
        DeepEquilibriumModel, UniversalTransformerBlock = _get_deq_agent()
        
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={'hidden_dim': self.hidden_dim, 'heads': self.heads, 'mlp_dim': self.mlp_dim},
            max_iter=self.deq_iters, tol=self.deq_tol, beta=self.deq_beta, m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


class RecurseZeroAgentFast(nn.Module):
    """Fast DEQ agent with minimal iterations."""
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_actions: int = 4672
    deq_iters: int = 2
    deq_tol: float = 0.1
    deq_beta: float = 0.9
    deq_m: int = 2
    
    @nn.compact
    def __call__(self, x):
        DeepEquilibriumModel, UniversalTransformerBlock = _get_deq_agent()
        
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={'hidden_dim': self.hidden_dim, 'heads': self.heads, 'mlp_dim': self.mlp_dim},
            max_iter=self.deq_iters, tol=self.deq_tol, beta=self.deq_beta, m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward
