import jax
import jax.numpy as jnp
import flax.linen as nn

class GTrXLGating(nn.Module):
    """
    Gated Transformer-XL (GTrXL) Gating Mechanism.
    
    Stabilizes deep or recursive architectures by replacing standard residual connections
    with a GRU-style gating update.
    
    Ref: https://arxiv.org/abs/1910.06764
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x, f_x):
        """
        Applies the GTrXL gating.
        
        Args:
            x: Input tensor (h_{t-1}) corresponding to the residual path.
            f_x: Transformed tensor (f(h_{t-1})) corresponding to the layer output.
            
        Returns:
            h_t: The updated state.
        """
        # We need linear projections for the GRU gates
        # The equation from text:
        # r_t = sigmoid(W_r h_{t-1} + U_r f(h_{t-1}))
        # z_t = sigmoid(W_z h_{t-1} + U_z f(h_{t-1}))
        # h_tilde = tanh(W_h (r_t * h_{t-1}) + U_h f(h_{t-1}))
        # h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        
        # We assume bias is included in the dense layers
        
        # Parameters for W (acting on h_{t-1})
        Wr = nn.Dense(self.hidden_dim, use_bias=False, name='Wr')(x)
        Wz = nn.Dense(self.hidden_dim, use_bias=False, name='Wz')(x)
        Wh = nn.Dense(self.hidden_dim, use_bias=False, name='Wh') # Applied later
        
        # Parameters for U (acting on f(h_{t-1}))
        Ur = nn.Dense(self.hidden_dim, name='Ur')(f_x)
        Uz = nn.Dense(self.hidden_dim, name='Uz')(f_x)
        Uh = nn.Dense(self.hidden_dim, name='Uh')(f_x)
        
        # Compute gates
        r_t = nn.sigmoid(Wr + Ur)
        z_t = nn.sigmoid(Wz + Uz)
        
        # Kandiate activation
        # W_h is applied to (r_t * h_{t-1})
        Wh_out = Wh(r_t * x)
        h_tilde = nn.tanh(Wh_out + Uh)
        
        # Final update
        h_t = (1 - z_t) * x + z_t * h_tilde
        
        return h_t
