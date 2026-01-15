"""
Int8 Quantization Module for RecurseZero.

CURRENTLY DISABLED BY DEFAULT due to:
1. AQT requires RNG keys that conflict with DEQ's fixed-point iteration
2. L4 GPU shows buffer comparator precision errors with Int8

To enable (experimental): set _FORCE_ENABLE = True

Uses JAX AQT (Accurate Quantized Training) to enable Int8 inference
for 2-4x speedup on Tensor Cores when working properly.
"""

import jax
import jax.numpy as jnp

# DISABLED by default - Int8 causes issues on some hardware
_FORCE_ENABLE = False
_AQT_AVAILABLE = False
_INT8_CONFIG = None


def _try_import_aqt():
    """Try to import AQT and create config."""
    global _AQT_AVAILABLE, _INT8_CONFIG
    
    if not _FORCE_ENABLE:
        _AQT_AVAILABLE = False
        _INT8_CONFIG = None
        return False
    
    try:
        import aqt.jax.v2.config as aqt_config
        import aqt.jax.v2.flax.aqt_flax as aqt_flax
        
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
    
    NOTE: Currently disabled due to hardware compatibility issues.
    Returns False and uses FP32 for stability.
    
    Args:
        enable: Whether to attempt enabling (currently ignored)
        
    Returns:
        True if successfully enabled (currently always False)
    """
    global _AQT_AVAILABLE, _FORCE_ENABLE
    
    # Quantization is disabled due to:
    # 1. AQT needs RNG in apply() which DEQ doesn't provide
    # 2. Buffer comparator errors on L4 GPU
    print("⚪ Int8 quantization disabled (using stable FP32)")
    print("   Note: AQT conflicts with DEQ's fixed-point iteration")
    _AQT_AVAILABLE = False
    return False


def get_dot_general():
    """
    Get a quantized dot_general for use in nn.Dense.
    
    Currently always returns None (FP32 mode).
    """
    return None  # Disabled


def is_enabled() -> bool:
    """Check if Int8 quantization is currently enabled."""
    return False  # Disabled
