import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias

# Import quantization - get_dot_general returns None if unavailable
try:
    from optimization.quantization import get_dot_general, is_enabled as quant_enabled
except ImportError:
    def get_dot_general():
        return None
    def quant_enabled():
        return False


class QuantizedDense(nn.Module):
    """
    Dense layer with optional Int8 quantization via dot_general injection.
    
    This is the correct way to use AQT with Flax - inject the quantized
    dot_general into the standard nn.Dense layer.
    """
    features: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x):
        dg = get_dot_general()
        
        if dg is not None:
            return nn.Dense(
                features=self.features,
                use_bias=self.use_bias,
                dot_general=dg
            )(x)
        else:
            return nn.Dense(
                features=self.features,
                use_bias=self.use_bias
            )(x)


class ChessformerAttention(nn.Module):
    """
    Multi-head attention with:
    - Int8 quantization on projections
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
        
        # QKV projections (quantized if enabled)
        q = QuantizedDense(total_dim, use_bias=False, name='query')(x)
        k = QuantizedDense(total_dim, use_bias=False, name='key')(x)
        v = QuantizedDense(total_dim, use_bias=False, name='value')(x)
        
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
        # Each head learns different patterns (e.g., "diagonal", "knight jump")
        if capture_attention:
            self.sow('intermediates', 'attention_weights', attn_weights)
            
            # Also store per-head summaries for quick analysis
            # Mean attention per head for interpretability
            mean_attn_per_head = jnp.mean(attn_weights, axis=(0, 2, 3))  # (H,)
            self.sow('intermediates', 'attention_entropy', 
                     -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-10), axis=-1).mean())
        
        # Apply attention to values
        out = jnp.matmul(attn_weights, v)
        
        # Reshape back to (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Output projection (quantized if enabled)
        out = QuantizedDense(D, use_bias=False, name='output')(out)
        
        return out


class UniversalTransformerBlock(nn.Module):
    """
    Single step of Universal Transformer / DEQ.
    
    Full spec compliance:
    - GTrXL gating for stability (spec 2.3)
    - Chess relative positional encodings (spec 2.4)
    - Int8 quantization via AQT (spec 4.1)
    - Attention visualization (spec 5.1)
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
        
        # 4. MLP Block with quantization
        h_mlp = nn.LayerNorm(name='norm2')(z_attn)
        
        mlp_out = QuantizedDense(self.mlp_dim, name='mlp_up')(h_mlp)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = QuantizedDense(self.hidden_dim, name='mlp_down')(mlp_out)
        
        # 5. Second GTrXL Gate
        gate2 = GTrXLGating(hidden_dim=self.hidden_dim, name='gate2')
        z_next = gate2(z_attn, mlp_out)
        
        return z_next


# Alias for backward compatibility
CustomAttention = ChessformerAttention
