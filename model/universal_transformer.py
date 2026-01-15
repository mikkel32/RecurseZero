import jax
import jax.numpy as jnp
import flax.linen as nn
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias

class CustomAttention(nn.Module):
    num_heads: int
    head_dim: int
    
    @nn.compact
    def __call__(self, x, bias=None):
        # x: (Batch, Seq, Dim)
        B, L, D = x.shape
        
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        # Reshape to (B, L, H, D_head) -> (B, H, L, D_head)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Dot product
        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        dist = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        dist = dist / jnp.sqrt(self.head_dim)
        
        if bias is not None:
            # Bias shape: (1, H, L, L)
            dist = dist + bias
            
        attn_weights = nn.softmax(dist, axis=-1)
        
        # SOW weights for visualization
        self.sow('intermediates', 'attn_weights', attn_weights)
        
        # Apply to V
        # (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        out = jnp.matmul(attn_weights, v)
        
        # Transpose back: (B, L, H, D) -> (B, L, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Final projection
        out = nn.Dense(D, use_bias=False)(out)
        
        return out

class UniversalTransformerBlock(nn.Module):
    """
    Single step of the Universal Transformer / DEQ.
    """
    hidden_dim: int
    heads: int
    mlp_dim: int
    
    def setup(self):
        self.norm1 = nn.LayerNorm()
        # Use custom attention to support sowing
        self.attn = CustomAttention(
            num_heads=self.heads,
            head_dim=self.hidden_dim // self.heads
        )
        self.gate1 = GTrXLGating(hidden_dim=self.hidden_dim)
        
        self.norm2 = nn.LayerNorm()
        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dense(self.hidden_dim)
        ])
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
        
        # Gate z (state) with update
        z_attn = self.gate1(z, attn_out)
        
        # 2. MLP Block
        h_mlp = self.norm2(z_attn)
        mlp_out = self.mlp(h_mlp)
        
        z_next = self.gate2(z_attn, mlp_out)
        
        return z_next
