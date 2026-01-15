"""
Universal Transformer Block for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec:
- Single block recycled for DEQ (section 2.1)
- GTrXL gating for stability (section 2.3)
- Chess-aware relative positional encodings (section 2.4)
- ChessformerAttention for visualization (section 5.1)
- BFloat16 mixed precision for speed
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias

# Get compute dtype from mixed precision settings
try:
    from optimization.mixed_precision import get_dtype, is_bf16_enabled
except ImportError:
    def get_dtype():
        return jnp.float32
    def is_bf16_enabled():
        return False


class ChessformerAttention(nn.Module):
    """
    Multi-head attention with:
    - BFloat16 compute for speed
    - Attention weight visualization (spec 5.1)
    - Chess-specific relative position bias
    """
    num_heads: int
    head_dim: int
    dtype: jnp.dtype = None  # Will use BF16 when enabled
    
    @nn.compact
    def __call__(self, x, bias=None, capture_attention: bool = True):
        B, L, D = x.shape
        total_dim = self.num_heads * self.head_dim
        
        # Use BF16 dtype if enabled
        compute_dtype = self.dtype if self.dtype is not None else get_dtype()
        
        # QKV projections with BF16
        q = nn.Dense(total_dim, use_bias=False, dtype=compute_dtype, name='query')(x)
        k = nn.Dense(total_dim, use_bias=False, dtype=compute_dtype, name='key')(x)
        v = nn.Dense(total_dim, use_bias=False, dtype=compute_dtype, name='value')(x)
        
        # Reshape to (B, H, L, D_head)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention (in BF16)
        scale = jnp.sqrt(jnp.array(self.head_dim, dtype=compute_dtype))
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        
        # Add relative position bias
        if bias is not None:
            attn_logits = attn_logits + bias.astype(compute_dtype)
            
        attn_weights = nn.softmax(attn_logits, axis=-1)
        
        # Visualization
        if capture_attention:
            self.sow('intermediates', 'attention_weights', attn_weights)
            self.sow('intermediates', 'attention_entropy', 
                     -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-10), axis=-1).mean())
        
        # Apply attention to values
        out = jnp.matmul(attn_weights, v)
        
        # Reshape back to (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Output projection
        out = nn.Dense(D, use_bias=False, dtype=compute_dtype, name='output')(out)
        
        return out


class UniversalTransformerBlock(nn.Module):
    """
    Single step of Universal Transformer / DEQ with BFloat16.
    
    Full spec compliance:
    - GTrXL gating for stability (spec 2.3)
    - Chess relative positional encodings (spec 2.4)
    - Attention visualization (spec 5.1)
    - BFloat16 mixed precision for 2x speedup
    """
    hidden_dim: int
    heads: int
    mlp_dim: int
    dtype: jnp.dtype = None  # Will use BF16 when enabled
    
    @nn.compact
    def __call__(self, z, x_input, train: bool = True):
        # Get compute dtype
        compute_dtype = self.dtype if self.dtype is not None else get_dtype()
        
        # 1. Input injection and Layer Norm
        h = z + x_input
        h_norm = nn.LayerNorm(dtype=compute_dtype, name='norm1')(h)
        
        # 2. Chess-aware Attention
        attn = ChessformerAttention(
            num_heads=self.heads,
            head_dim=self.hidden_dim // self.heads,
            dtype=compute_dtype,
            name='attention'
        )
        
        # Generate chess-specific positional bias
        seq_len = h.shape[1]
        rel_pos_bias = ChessRelativePositionBias(
            num_heads=self.heads, 
            name='chess_pos_bias'
        )
        attn_bias = rel_pos_bias(seq_len, seq_len)
        
        attn_out = attn(h_norm, bias=attn_bias, capture_attention=True)
        
        # 3. GTrXL Gating (spec 2.3)
        gate1 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate1')
        z_attn = gate1(z, attn_out)
        
        # 4. MLP Block with BF16
        h_mlp = nn.LayerNorm(dtype=compute_dtype, name='norm2')(z_attn)
        
        mlp_out = nn.Dense(self.mlp_dim, dtype=compute_dtype, name='mlp_up')(h_mlp)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='mlp_down')(mlp_out)
        
        # 5. Second GTrXL Gate
        gate2 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate2')
        z_next = gate2(z_attn, mlp_out)
        
        return z_next


# Alias for backward compatibility
CustomAttention = ChessformerAttention
