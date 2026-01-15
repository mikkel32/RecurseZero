"""
Mixed Precision Training for RecurseZero.

BFloat16 provides:
- 2x memory reduction vs FP32
- 1.5-2x speedup on Tensor Cores
- Native JAX support

IMPORTANT: Call init_mixed_precision() BEFORE importing models!
"""

import jax
import jax.numpy as jnp

# Global precision setting - mutable at runtime
_GLOBAL_DTYPE = jnp.float32  # Default to FP32


def init_mixed_precision(enable: bool = True) -> bool:
    """
    Enable BFloat16 mixed precision training.
    
    MUST be called before model initialization!
    
    Args:
        enable: Whether to enable BF16
        
    Returns:
        True if enabled successfully
    """
    global _GLOBAL_DTYPE
    
    if not enable:
        _GLOBAL_DTYPE = jnp.float32
        print("⚪ Mixed precision disabled (FP32)")
        return False
    
    backend = jax.default_backend()
    
    if backend in ('gpu', 'tpu'):
        _GLOBAL_DTYPE = jnp.bfloat16
        print(f"✓ BFloat16 enabled (dtype={_GLOBAL_DTYPE})")
        return True
    else:
        _GLOBAL_DTYPE = jnp.float32
        print(f"⚪ BFloat16 not supported on {backend}, using FP32")
        return False


def get_dtype():
    """
    Get the current compute dtype.
    
    This is called at RUNTIME to get the correct dtype.
    """
    global _GLOBAL_DTYPE
    return _GLOBAL_DTYPE


def is_bf16_enabled() -> bool:
    """Check if BF16 is currently enabled."""
    global _GLOBAL_DTYPE
    return _GLOBAL_DTYPE == jnp.bfloat16


def get_param_dtype():
    """Get dtype for parameter storage."""
    return jnp.float32  # Always FP32 for stability


def cast_if_needed(x, target_dtype=None):
    """Cast tensor to target dtype if different."""
    if target_dtype is None:
        target_dtype = get_dtype()
    if x.dtype != target_dtype:
        return x.astype(target_dtype)
    return x
