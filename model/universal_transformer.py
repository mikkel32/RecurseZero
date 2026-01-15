import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias

# Try to import quantization, fallback gracefully
try:
    from optimization.quantization import QuantizedDense, is_quantization_enabled
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    def is_quantization_enabled():
        return False

class CustomAttention(nn.Module):
    """
    Multi-head attention with optional Int8 quantization on projections.
    """
    num_heads: int
    head_dim: int
    
    @nn.compact
    def __call__(self, x, bias=None):
        B, L, D = x.shape
        
        # Use quantized Dense if available
        if HAS_QUANTIZATION and is_quantization_enabled():
            q = QuantizedDense(self.num_heads * self.head_dim, use_bias=False)(x)
            k = QuantizedDense(self.num_heads * self.head_dim, use_bias=False)(x)
            v = QuantizedDense(self.num_heads * self.head_dim, use_bias=False)(x)
        else:
            q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
            k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
            v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        # Reshape to (B, L, H, D_head) -> (B, H, L, D_head)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        dist = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        dist = dist / jnp.sqrt(self.head_dim)
        
        if bias is not None:
            dist = dist + bias
            
        attn_weights = nn.softmax(dist, axis=-1)
        
        # Store attention for visualization
        self.sow('intermediates', 'attn_weights', attn_weights)
        
        # Apply to V
        out = jnp.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Output projection (quantized if enabled)
        if HAS_QUANTIZATION and is_quantization_enabled():
            out = QuantizedDense(D, use_bias=False)(out)
        else:
            out = nn.Dense(D, use_bias=False)(out)
        
        return out

class UniversalTransformerBlock(nn.Module):
    """
    Single step of Universal Transformer / DEQ with Int8 quantization.
    
    The MLP uses Int8 matmuls when AQT is enabled, providing 2-4x speedup
    on Tensor Cores while maintaining model quality.
    """
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn = CustomAttention(
            num_heads=self.heads,
            head_dim=self.hidden_dim // self.heads
        )
        self.gate1 = GTrXLGating(hidden_dim=self.hidden_dim)
        
        self.norm2 = nn.LayerNorm()
        self.gate2 = GTrXLGating(hidden_dim=self.hidden_dim)
        
        self.rel_pos_bias = ChessRelativePositionBias(num_heads=self.heads)

    def __call__(self, z, x_input, train: bool = True):
        # Input Injection
        h = z + x_input
        
        # 1. Attention Block
        h_norm = self.norm1(h)
        seq_len = h.shape[1]
        attn_bias = self.rel_pos_bias(seq_len, seq_len)
        
        attn_out = self.attn(h_norm, bias=attn_bias)
        
        # GTrXL gating
        z_attn = self.gate1(z, attn_out)
        
        # 2. MLP Block (Quantized if enabled)
        h_mlp = self.norm2(z_attn)
        
        if HAS_QUANTIZATION and is_quantization_enabled():
            mlp_out = QuantizedDense(self.mlp_dim)(h_mlp)
            mlp_out = nn.gelu(mlp_out)
            mlp_out = QuantizedDense(self.hidden_dim)(mlp_out)
        else:
            mlp_out = nn.Dense(self.mlp_dim)(h_mlp)
            mlp_out = nn.gelu(mlp_out)
            mlp_out = nn.Dense(self.hidden_dim)(mlp_out)
        
        z_next = self.gate2(z_attn, mlp_out)
        
        return z_next
