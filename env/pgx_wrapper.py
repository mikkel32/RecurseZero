import jax
import jax.numpy as jnp
import pgx
from functools import partial

class RecurseEnv:
    """
    A GPU-Resident JAX environment wrapper using Pgx.
    
    This class handles:
    - Environment stepping on device (GPU/MPS).
    - Auto-vectorization for massive parallelism.
    - State management without CPU transfer.
    """
    def __init__(self, batch_size: int = 4096, seed: int = 0):
        self.batch_size = batch_size
        self.env_id = "chess"
        self.env = pgx.make(self.env_id)
        self.key = jax.random.PRNGKey(seed)
        
        # JIT-compile core interactions (Pure GPU Resident)
        self._init = jax.jit(jax.vmap(self.env.init))
        self._step = jax.jit(jax.vmap(self.env.step))

    def init(self, key):
        """
        Initialize the parallel environments.
        Returns:
            State: The initial state of the environments on the GPU.
        """
        keys = jax.random.split(key, self.batch_size)
        return self._init(keys)

    def step(self, state, actions):
        """
        Execute actions in parallel environments.
        
        Args:
            state: The current environment state (on GPU).
            actions: The actions to execute (on GPU).
            
        Returns:
            next_state: The resulting state (on GPU).
        """
        return self._step(state, actions)

    @property
    def num_actions(self):
        # Hardcoding for Chess to avoid JIT compilation overlap/Metal issues during startup
        return 4672
        # return self.env.num_actions

    @property
    def observation_shape(self):
        # Hardcoding to avoid startup JIT issues on Metal
        return (8, 8, 119)
        # return self.env.observation_shape
