"""
Gated Transformer-XL (GTrXL) Gating Mechanism.

Per GPU ONLY Chess RL Agent.txt spec section 2.3:
Stabilizes deep or recursive architectures by replacing standard 
residual connections with a GRU-style gating update.

Ref: https://arxiv.org/abs/1910.06764
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

# Get compute dtype
try:
    from optimization.mixed_precision import get_dtype
except ImportError:
    def get_dtype():
        return jnp.float32


class GTrXLGating(nn.Module):
    """
    GTrXL Gating with BFloat16 support.
    
    The gating mechanism:
    r_t = σ(W_r h_{t-1} + U_r f(h_{t-1}))
    z_t = σ(W_z h_{t-1} + U_z f(h_{t-1}))
    h̃_t = tanh(W_h (r_t ⊙ h_{t-1}) + U_h f(h_{t-1}))
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """
    hidden_dim: int
    dtype: jnp.dtype = None  # Uses BF16 when enabled
    
    @nn.compact
    def __call__(self, x, f_x):
        """
        Applies the GTrXL gating.
        
        Args:
            x: Input tensor (h_{t-1}) - residual path
            f_x: Transformed tensor f(h_{t-1}) - layer output
            
        Returns:
            h_t: The updated state
        """
        compute_dtype = self.dtype if self.dtype is not None else get_dtype()
        
        # Parameters for W (acting on h_{t-1})
        Wr = nn.Dense(self.hidden_dim, use_bias=False, dtype=compute_dtype, name='Wr')(x)
        Wz = nn.Dense(self.hidden_dim, use_bias=False, dtype=compute_dtype, name='Wz')(x)
        Wh = nn.Dense(self.hidden_dim, use_bias=False, dtype=compute_dtype, name='Wh')
        
        # Parameters for U (acting on f(h_{t-1}))
        Ur = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='Ur')(f_x)
        Uz = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='Uz')(f_x)
        Uh = nn.Dense(self.hidden_dim, dtype=compute_dtype, name='Uh')(f_x)
        
        # Compute gates
        r_t = nn.sigmoid(Wr + Ur)
        z_t = nn.sigmoid(Wz + Uz)
        
        # Candidate activation
        Wh_out = Wh(r_t * x)
        h_tilde = nn.tanh(Wh_out + Uh)
        
        # Final update
        h_t = (1 - z_t) * x + z_t * h_tilde
        
        return h_t
