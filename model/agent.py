import jax
import jax.numpy as jnp
import flax.linen as nn
from .deq import DeepEquilibriumModel
from .universal_transformer import UniversalTransformerBlock
from algorithm.muesli import PolicyHead
from algorithm.pve import ValueHead, RewardHead

class RecurseZeroAgent(nn.Module):
    """
    The full RecurseZero Agent.
    
    Per GPU ONLY Chess RL Agent.txt spec:
    - DEQ core with Anderson Acceleration
    - GTrXL stabilization
    - Chess-aware positional encodings
    - Muesli policy head (search-free)
    - PVE value/reward heads
    - Int8 quantization (when enabled)
    
    Structure:
    Input (Board) -> Embedding -> DEQ -> Abstract State z*
    z* -> Policy Head -> Logits
    z* -> Value Head -> V(s)
    z* -> Reward Head -> R(s)
    
    Complexity metrics available via mutable='intermediates':
    - final_residual: Per-sample DEQ convergence
    - attention_weights: Per-head attention maps
    - attention_entropy: Measure of attention confidence
    """
    # SPEC-COMPLIANT DEFAULTS
    hidden_dim: int = 256  # Full size per spec
    heads: int = 8         # More heads for chess patterns
    mlp_dim: int = 1024    # Full MLP size
    num_actions: int = 4672
    
    # DEQ PARAMETERS
    deq_iters: int = 8     # Increased for better convergence
    deq_tol: float = 1e-3  # Early exit tolerance
    deq_beta: float = 0.9  # Anderson damping
    deq_m: int = 3         # History size (2x2 solve)
    
    @nn.compact
    def __call__(self, x):
        # x: Input observation using Pgx encoding (Batch, 8, 8, C)
        
        # 1. Input Embedding
        # Project 8x8xC to 8x8xDim
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        
        # Flatten for Transformer: (Batch, 64, Dim)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        # 2. DEQ Core
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        
        # 3. Output Heads
        # Mean pooling over the 64 squares
        z_pooled = jnp.mean(z_star, axis=1)  # (Batch, Dim)
        
        # Policy: Direct move probabilities (Muesli, search-free)
        policy_logits = PolicyHead(
            self.hidden_dim, 
            self.num_actions,
            name='policy'
        )(z_pooled)
        
        # Value: Win probability prediction
        value = ValueHead(
            self.hidden_dim,
            name='value'
        )(z_pooled)
        
        # Reward: Immediate reward prediction (PVE)
        reward = RewardHead(
            self.hidden_dim,
            name='reward'
        )(z_pooled)
        
        return policy_logits, value, reward


class RecurseZeroAgentFast(nn.Module):
    """
    Speed-optimized version for L4 GPU (24GB).
    
    Reduced parameters for faster training while maintaining
    the core architecture.
    """
    hidden_dim: int = 128
    heads: int = 4
    mlp_dim: int = 512
    num_actions: int = 4672
    deq_iters: int = 4
    deq_tol: float = 1e-2
    deq_beta: float = 0.9
    deq_m: int = 2
    
    @nn.compact
    def __call__(self, x):
        x_embed = nn.Dense(self.hidden_dim, name='input_embed')(x)
        x_flat = x_embed.reshape(x_embed.shape[0], -1, self.hidden_dim)
        
        deq = DeepEquilibriumModel(
            block_class=UniversalTransformerBlock,
            block_args={
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'mlp_dim': self.mlp_dim
            },
            max_iter=self.deq_iters,
            tol=self.deq_tol,
            beta=self.deq_beta,
            m=self.deq_m,
            name='deq'
        )
        
        z_star = deq(x_flat)
        z_pooled = jnp.mean(z_star, axis=1)
        
        policy_logits = PolicyHead(self.hidden_dim, self.num_actions, name='policy')(z_pooled)
        value = ValueHead(self.hidden_dim, name='value')(z_pooled)
        reward = RewardHead(self.hidden_dim, name='reward')(z_pooled)
        
        return policy_logits, value, reward
