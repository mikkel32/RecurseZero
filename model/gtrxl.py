"""
GTrXL Gating Mechanism for stable recursive networks.

Per spec section 2.3: Stabilizes deep/recursive architectures
by replacing residual connections with GRU-style gating.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class GTrXLGating(nn.Module):
    """GTrXL Gating for stable DEQ convergence."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x, f_x):
        # W projections (acting on x)
        Wr = nn.Dense(self.hidden_dim, use_bias=False, name='Wr')(x)
        Wz = nn.Dense(self.hidden_dim, use_bias=False, name='Wz')(x)
        Wh = nn.Dense(self.hidden_dim, use_bias=False, name='Wh')
        
        # U projections (acting on f(x))
        Ur = nn.Dense(self.hidden_dim, name='Ur')(f_x)
        Uz = nn.Dense(self.hidden_dim, name='Uz')(f_x)
        Uh = nn.Dense(self.hidden_dim, name='Uh')(f_x)
        
        # Gates
        r_t = nn.sigmoid(Wr + Ur)
        z_t = nn.sigmoid(Wz + Uz)
        
        # Candidate
        Wh_out = Wh(r_t * x)
        h_tilde = nn.tanh(Wh_out + Uh)
        
        # Update
        h_t = (1 - z_t) * x + z_t * h_tilde
        
        return h_t
