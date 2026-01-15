"""
RecurseZero Agent - GPU-Only Spec Compliant.

Per GPU-Only Chess RL Agent.txt spec:
- Section 2.1: Universal Transformer (weight recycling)
- Section 2.2: DEQ with Anderson Acceleration (infinite depth)
- Section 2.2.2: Implicit differentiation (O(1) memory)
- Section 2.3: GTrXL gating (stability)
- Section 2.4: Chess-aware positional encodings
- Section 3: Muesli policy optimization (search-free)
- Section 3.3: PVE value/reward heads

ZERO CPU: All operations compile to XLA GPU kernels.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-ONLY SIMPLE TRANSFORMER (Fast Training)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTransformerBlock(nn.Module):
    """
    Standard transformer block for fast GPU training.
    
    100% XLA-compiled, zero CPU operations.
    """
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape
        head_dim = self.hidden_dim // self.heads
        
        # 1. LayerNorm + Multi-head attention
        h = nn.LayerNorm()(x)
        
        # QKV projection (fused for efficiency)
        qkv = nn.Dense(3 * self.hidden_dim, use_bias=False)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        q = q.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention (pure JAX)
        scale = jnp.sqrt(jnp.float32(head_dim))
        attn_weights = nn.softmax(jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)
        
        # Merge heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_dim)
        attn_out = nn.Dense(self.hidden_dim)(attn_out)
        
        x = x + attn_out
        
        # 2. MLP block
        h = nn.LayerNorm()(x)
        mlp = nn.Dense(self.mlp_dim)(h)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.hidden_dim)(mlp)
        
        return x + mlp


class RecurseZeroAgentSimple(nn.Module):
    """
    Fast GPU-only agent without DEQ.
    
    For training speed. Uses stacked transformer layers.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_layers: int = 3
    num_actions: int = 4672
    
    @nn.compact
    def __call__(self, x):
        # 1. Embedding
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. Stacked transformer blocks
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
# SPEC-COMPLIANT DEQ AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_deq_components():
    """Import DEQ components (lazy to avoid circular imports)."""
    from .deq import DeepEquilibriumModel
    from .universal_transformer import UniversalTransformerBlock
    return DeepEquilibriumModel, UniversalTransformerBlock


class RecurseZeroAgent(nn.Module):
    """
    Full spec-compliant DEQ agent.
    
    Per GPU-Only Chess RL Agent.txt:
    - Section 2.1: 256-dim Universal Transformer
    - Section 2.2: 8 DEQ iterations (infinite depth)
    - Section 2.3: GTrXL gating for stability
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
        DeepEquilibriumModel, UniversalTransformerBlock = _get_deq_components()
        
        # 1. Input embedding
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core (infinite depth, O(1) memory)
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
        
        # 3. Output heads (Muesli policy + PVE value)
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


class RecurseZeroAgentFast(nn.Module):
    """
    Fast DEQ agent - balanced speed and spec compliance.
    
    Reduced iterations but still uses DEQ architecture.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_actions: int = 4672
    deq_iters: int = 3
    deq_tol: float = 1e-2
    deq_beta: float = 0.9
    deq_m: int = 2
    
    @nn.compact
    def __call__(self, x):
        DeepEquilibriumModel, UniversalTransformerBlock = _get_deq_components()
        
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
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
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward


# ═══════════════════════════════════════════════════════════════════════════════
# LICHESS SUPERVISED AGENT (Same architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LichessAgent(nn.Module):
    """
    Agent for supervised learning on Lichess data.
    
    Same architecture as RecurseZeroAgentSimple, but can be
    trained with different objective (cross-entropy vs RL).
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_layers: int = 3
    num_actions: int = 4672
    
    @nn.compact
    def __call__(self, x):
        # Use same architecture as Simple agent
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        h = x_flat
        for i in range(self.num_layers):
            h = SimpleTransformerBlock(
                hidden_dim=self.hidden_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                name=f'block_{i}'
            )(h)
        
        z_pooled = jnp.mean(h, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward
