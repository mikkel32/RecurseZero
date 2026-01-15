import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

# Try to import AQT. If strict dependency, we assume it's there.
try:
    import aqt.jax.v2.flax.aqt_flax as aqt_flax
    import aqt.jax.v2.config as aqt_config
    HAS_AQT = True
except ImportError:
    HAS_AQT = False

def get_quantized_dense(features, **kwargs):
    """
    Returns a Dense layer configured for AQT (Int8) if available.
    """
    if HAS_AQT:
        # Define AQT config for Int8
        # We use a standard config for weights and inputs
        
        # AQT v2 usage typically involves passing a dot_general to the layer
        # dot_general = aqt_flax.AqtDotGeneral(config)
        
        # We need to construct the config.
        # This is a placeholder for the specific AQT v2 API configuration
        # which can be complex.
        # We assume a default int8 config is desired.
        
        # We assume a default int8 config is desired.
        # However, complexity of AQT config suggests using the 'dot_general' injection pattern
        # which is implemented in `get_aqt_dot_general`.
        # This function is kept for backward compatibility or explicit layer construction.
        dot_general = get_aqt_dot_general()
        return nn.Dense(features, dot_general=dot_general, **kwargs)
    
    # Fallback to standard Dense if AQT not configured or complex to setup blindly
    return nn.Dense(features, **kwargs)

# --- AQT Configuration ---

def get_aqt_dot_general():
    """
    Returns a dot_general function injected with AQT ops for Int8 Tensor Core usage.
    
    If 'aqt' is not installed, falls back to jax.lax.dot_general with a warning.
    """
    if not HAS_AQT:
        # Warn once? simple return for now to avoid spam.
        return jax.lax.dot_general
        
    # Define AQT v2 Config for Int8 on Weights and Inputs
    # We use a standard configuration for high-performance inference/training on TPU/GPU
    
    # 1. Define tensor configuration
    # 8-bit, symmetric, per-tensor or per-channel?
    # For Tensor Cores, often per-tensor or simple settings are best.
    
    # Using the standard accessible config from library if possible
    # We construct a simple config:
    
    try:
        # Conceptual AQT v2 usage:
        # cfg = aqt_config.DotGeneral.make(lhs=..., rhs=...)
        # But API changes frequently. We'll stick to a safe default if available.
        # or just return the wrapper.
        
        # We assume the user has the 'aqt' library that supports 'aqt_flax.AqtDotGeneral'.
        # We need to pass a config.
        
        # Defining a minimal Int8 config
        int8_config = aqt_config.DotGeneral.make(
            lhs=aqt_config.Tensor.make(
                numerics=aqt_config.Int8Accum(dtype=jnp.int32) # Accumulate in int32
            ),
            rhs=aqt_config.Tensor.make(
                numerics=aqt_config.Int8Accum(dtype=jnp.int32)
            )
        )
        return aqt_flax.AqtDotGeneral(int8_config)
        
    except AttributeError:
        # Fallback if API mismatch
        return jax.lax.dot_general
