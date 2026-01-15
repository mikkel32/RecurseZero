"""
Universal Transformer Block for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec:
- Single block recycled for DEQ (section 2.1)
- GTrXL gating for stability (section 2.3)
- Chess-aware relative positional encodings (section 2.4)
- ChessformerAttention for visualization (section 5.1)

Simple FP32 implementation for stability.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias


class ChessformerAttention(nn.Module):
    """Multi-head attention with chess-aware positional bias."""
    num_heads: int
    head_dim: int
    
    @nn.compact
    def __call__(self, x, bias=None, capture_attention: bool = True):
        B, L, D = x.shape
        total_dim = self.num_heads * self.head_dim
        
        # QKV projections
        q = nn.Dense(total_dim, use_bias=False, name='query')(x)
        k = nn.Dense(total_dim, use_bias=False, name='key')(x)
        v = nn.Dense(total_dim, use_bias=False, name='value')(x)
        
        # Reshape to (B, H, L, D_head)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.float32(self.head_dim))
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        
        # Add relative position bias
        if bias is not None:
            attn_logits = attn_logits + bias
            
        attn_weights = nn.softmax(attn_logits, axis=-1)
        
        # Visualization
        if capture_attention:
            self.sow('intermediates', 'attention_weights', attn_weights)
        
        # Apply attention to values
        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = nn.Dense(D, use_bias=False, name='output')(out)
        
        return out


class UniversalTransformerBlock(nn.Module):
    """Single step of Universal Transformer / DEQ."""
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    @nn.compact
    def __call__(self, z, x_input, train: bool = True):
        # 1. Input injection + Layer Norm
        h = z + x_input
        h_norm = nn.LayerNorm(name='norm1')(h)
        
        # 2. Chess-aware Attention
        attn = ChessformerAttention(
            num_heads=self.heads,
            head_dim=self.hidden_dim // self.heads,
            name='attention'
        )
        
        # Positional bias
        seq_len = h.shape[1]
        rel_pos_bias = ChessRelativePositionBias(
            num_heads=self.heads, 
            name='chess_pos_bias'
        )
        attn_bias = rel_pos_bias(seq_len, seq_len)
        
        attn_out = attn(h_norm, bias=attn_bias)
        
        # 3. GTrXL Gating
        gate1 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate1')
        z_attn = gate1(z, attn_out)
        
        # 4. MLP Block
        h_mlp = nn.LayerNorm(name='norm2')(z_attn)
        mlp_out = nn.Dense(self.mlp_dim, name='mlp_up')(h_mlp)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.hidden_dim, name='mlp_down')(mlp_out)
        
        # 5. Second GTrXL Gate
        gate2 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate2')
        z_next = gate2(z_attn, mlp_out)
        
        return z_next


CustomAttention = ChessformerAttention
