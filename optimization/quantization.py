"""
Int8 Quantization Module for RecurseZero.

Uses JAX AQT (Accurate Quantized Training) to enable Int8 inference
for 2-4x speedup on Tensor Cores.

The key is injecting quantized dot_general into Flax Dense layers.
"""

import jax
import jax.numpy as jnp

# Global state
_AQT_AVAILABLE = False
_INT8_CONFIG = None

def _try_import_aqt():
    """Try to import AQT and create config."""
    global _AQT_AVAILABLE, _INT8_CONFIG
    
    try:
        import aqt.jax.v2.config as aqt_config
        import aqt.jax.v2.flax.aqt_flax as aqt_flax
        
        # Create Int8 config
        _INT8_CONFIG = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
        _AQT_AVAILABLE = True
        return True
    except ImportError:
        _AQT_AVAILABLE = False
        _INT8_CONFIG = None
        return False

def init_quantization(enable: bool = True) -> bool:
    """
    Initialize Int8 quantization.
    
    Returns True if successfully enabled.
    """
    global _AQT_AVAILABLE
    
    if not enable:
        _AQT_AVAILABLE = False
        print("⚪ Int8 quantization disabled")
        return False
    
    success = _try_import_aqt()
    if success:
        print("✓ Int8 Quantization enabled (AQT)")
    else:
        print("⚠ AQT not available - install with: pip install aqtp")
    
    return success

def get_dot_general():
    """
    Get a quantized dot_general for use in nn.Dense.
    
    Returns None if quantization is not available/enabled.
    
    Usage:
        nn.Dense(features=256, dot_general=get_dot_general())(x)
    """
    global _AQT_AVAILABLE, _INT8_CONFIG
    
    if not _AQT_AVAILABLE or _INT8_CONFIG is None:
        return None
    
    try:
        import aqt.jax.v2.flax.aqt_flax as aqt_flax
        return aqt_flax.AqtDotGeneral(_INT8_CONFIG)
    except Exception:
        return None

def is_enabled() -> bool:
    """Check if Int8 quantization is currently enabled."""
    return _AQT_AVAILABLE and _INT8_CONFIG is not None


# Try to initialize on import (optional pre-check)
_try_import_aqt()
