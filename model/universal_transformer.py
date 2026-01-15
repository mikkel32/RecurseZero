"""
Universal Transformer Block for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec:
- Single block recycled for DEQ (section 2.1)
- GTrXL gating for stability (section 2.3)
- Chess-aware relative positional encodings (section 2.4)
- ChessformerAttention for visualization (section 5.1)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias


class ChessformerAttention(nn.Module):
    """
    Multi-head attention with:
    - Attention weight visualization (spec 5.1)
    - Chess-specific relative position bias
    
    Per spec section 5.1 (Chessformer Attention Maps):
    The attention weights are captured and can be visualized to show
    which squares the model is "attending to" for each head.
    """
    num_heads: int
    head_dim: int
    
    @nn.compact
    def __call__(self, x, bias=None, capture_attention: bool = True):
        B, L, D = x.shape
        total_dim = self.num_heads * self.head_dim
        
        # QKV projections (standard FP32 for stability)
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
        
        # Add relative position bias (chess geometry)
        if bias is not None:
            attn_logits = attn_logits + bias
            
        attn_weights = nn.softmax(attn_logits, axis=-1)
        
        # VISUALIZATION: Store attention weights per head (spec 5.1)
        # Shape: (B, H, L, L) = (Batch, Heads, 64, 64) for chess
        if capture_attention:
            self.sow('intermediates', 'attention_weights', attn_weights)
            # Entropy for confidence tracking
            self.sow('intermediates', 'attention_entropy', 
                     -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-10), axis=-1).mean())
        
        # Apply attention to values
        out = jnp.matmul(attn_weights, v)
        
        # Reshape back to (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Output projection
        out = nn.Dense(D, use_bias=False, name='output')(out)
        
        return out


class UniversalTransformerBlock(nn.Module):
    """
    Single step of Universal Transformer / DEQ.
    
    Full spec compliance:
    - GTrXL gating for stability (spec 2.3)
    - Chess relative positional encodings (spec 2.4)
    - Attention visualization (spec 5.1)
    
    Note: Int8 quantization disabled due to AQT/DEQ incompatibility.
    Uses stable FP32 for all operations.
    """
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    @nn.compact
    def __call__(self, z, x_input, train: bool = True):
        # 1. Layer Normalization (Pre-Norm)
        h = z + x_input  # Input injection
        h_norm = nn.LayerNorm(name='norm1')(h)
        
        # 2. Chess-aware Attention with visualization
        attn = ChessformerAttention(
            num_heads=self.heads,
            head_dim=self.hidden_dim // self.heads,
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
        
        # 4. MLP Block (standard FP32)
        h_mlp = nn.LayerNorm(name='norm2')(z_attn)
        
        mlp_out = nn.Dense(self.mlp_dim, name='mlp_up')(h_mlp)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.hidden_dim, name='mlp_down')(mlp_out)
        
        # 5. Second GTrXL Gate
        gate2 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate2')
        z_next = gate2(z_attn, mlp_out)
        
        return z_next


# Alias for backward compatibility
CustomAttention = ChessformerAttention
