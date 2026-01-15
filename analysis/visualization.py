import jax
import jax.numpy as jnp
import flax.linen as nn

def extract_attention_maps(agent_params, x_input, agent_config=None):
    """
    Runs the agent components to extract 'sowed' attention weights.
    
    Since the DEQ solver in the Agent discards intermediate states (to enforce O(1) memory),
    we must:
    1. Re-run the DEQ solver to find the Fixed Point z*.
    2. Run the Universal Transformer Block one last time on z* to capture the 'intermediates'.
    
    Args:
        agent_params: Parameter dictionary of the trained RecurseZeroAgent.
        x_input: Board observation (Batch, 8, 8, C).
        agent_config: Optional config dict to instantiate Agent/Block if defaults differ.
        
    Returns:
        accumulated_intermediates: The dictionary containing 'attn_weights'.
    """
    # 1. Instantiate the Block (stateless definition)
    # We need to mirror the structure in RecurseZeroAgent
    from model.universal_transformer import UniversalTransformerBlock
    from model.deq import fixed_point_iteration
    
    # Defaults from Agent
    hidden_dim = 256
    heads = 4
    mlp_dim = 1024
    if agent_config:
        hidden_dim = agent_config.get('hidden_dim', hidden_dim)
        heads = agent_config.get('heads', heads)
        mlp_dim = agent_config.get('mlp_dim', mlp_dim)
        
    block = UniversalTransformerBlock(hidden_dim=hidden_dim, heads=heads, mlp_dim=mlp_dim)
    
    # 2. Extract Block Parameters
    # Hierarchy: RecurseZeroAgent -> DeepEquilibriumModel_0 -> block_params
    # Flax implicitly names the DEQ module. We assume 'DeepEquilibriumModel_0' 
    # or we try to find it.
    
    deq_params = None
    # Try standard key
    if 'DeepEquilibriumModel_0' in agent_params['params']:
        deq_params = agent_params['params']['DeepEquilibriumModel_0']
    else:
        # Fallback search or assume flat if passed directly??
        raise ValueError("Could not find DeepEquilibriumModel parameters in agent_params")
        
    block_params = deq_params['block_params']
    
    # 3. Embedding (Re-run embedding layer)
    # Params: 'Dense_0' usually for embedding in Agent
    embed_params = agent_params['params']['Dense_0']
    
    # Re-implement embedding logic from Agent
    # x_embed = nn.Dense(hidden_dim)(x)
    # We use functional apply
    x_embed = nn.Dense(hidden_dim).apply({'params': embed_params}, x_input)
    x_flat = x_embed.reshape(x_embed.shape[0], -1, hidden_dim)
    
    # 4. Find Fixed Point z*
    # We rely on the pure function of the block
    def apply_fn(z):
        return block.apply({'params': block_params}, z, x_flat, train=False) # Train=False for vis?
    
    z_init = jnp.zeros_like(x_flat)
    
    # We re-run the solver (deterministically handling z*)
    solver_args = {'max_iter': 25, 'tol': 1e-4, 'alpha': 1.0}
    z_star = fixed_point_iteration(apply_fn, z_init, **solver_args)
    
    # 5. Capture Weights
    # Run block once with mutable=['intermediates']
    _, state = block.apply({'params': block_params}, z_star, x_flat, train=False, mutable=['intermediates'])
    
    return state['intermediates']
