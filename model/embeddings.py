import jax
import jax.numpy as jnp
import flax.linen as nn

class ChessRelativePositionBias(nn.Module):
    """
    2D Relative Positional Encoding for Chess (Chessformer).
    
    Encodes the spatial relationship between any two squares on the 8x8 board.
    Instead of absolute positions, we learn embeddings for the relative distance
    in rank and file (dx, dy).
    """
    num_heads: int
    
    @nn.compact
    def __call__(self, q_len, k_len):
        """
        Computes the relative position bias for attention.
        
        Args:
            q_len: sequence length of query (usually 64 for 8x8 squares)
            k_len: sequence length of key (usually 64)
            
        Returns:
            bias: tensor of shape (1, num_heads, q_len, k_len)
        """
        # We assume the sequence corresponds to the flattened 8x8 board (rank-major or file-major)
        # 0..63
        
        # Create a grid of coordinates
        coords = jnp.arange(64)
        x = coords // 8  # rank
        y = coords % 8   # file
        
        # Compute relative differences [q_len, k_len]
        # x[:, None] is (64, 1), x[None, :] is (1, 64) -> via broadcasting we get diffs
        x_diff = x[:, None] - x[None, :]  # Range: -7 to 7
        y_diff = y[:, None] - y[None, :]  # Range: -7 to 7
        
        # Shift to non-negative integers for embedding lookup using offset 7
        # 0..14
        x_indices = x_diff + 7
        y_indices = y_diff + 7
        
        # Define learnable embedding tables
        # We have 15 possible relative ranks (-7..7) and 15 relative files
        # We learn specific bias for each head
        
        # Shape: (15, num_heads)
        row_bias_table = self.param(
            'rel_pos_row_bias',
            nn.initializers.normal(stddev=0.02),
            (15, self.num_heads)
        )
        
        col_bias_table = self.param(
            'rel_pos_col_bias',
            nn.initializers.normal(stddev=0.02),
            (15, self.num_heads)
        )
        
        # Look up values
        # x_indices shape: (64, 64)
        # row_bias shape: (64, 64, num_heads)
        row_bias = row_bias_table[x_indices] 
        col_bias = col_bias_table[y_indices]
        
        # Combine biases (summing is a common strategy for disentangled att)
        # Shape: (64, 64, num_heads)
        total_bias = row_bias + col_bias
        
        # Permute to (1, num_heads, 64, 64) for attention broadcasting
        total_bias = jnp.transpose(total_bias, (2, 0, 1))
        
        return jnp.expand_dims(total_bias, 0)
