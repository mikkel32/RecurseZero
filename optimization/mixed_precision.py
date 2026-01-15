"""
Mixed Precision Training for RecurseZero.

BFloat16 provides:
- 2x memory reduction vs FP32
- 1.5-2x speedup on Tensor Cores
- Better numerical stability than FP16
- Native JAX support (no external libraries)

Per GPU ONLY Chess RL Agent spec section 4.1:
Uses mixed precision to overcome 8GB VRAM barrier.
"""

import jax
import jax.numpy as jnp
from functools import partial


# Global precision setting
_USE_BFLOAT16 = False


def init_mixed_precision(enable: bool = True) -> bool:
    """
    Enable BFloat16 mixed precision training.
    
    Args:
        enable: Whether to enable BF16
        
    Returns:
        True if enabled successfully
    """
    global _USE_BFLOAT16
    
    if not enable:
        _USE_BFLOAT16 = False
        print("⚪ Mixed precision disabled (FP32)")
        return False
    
    # Check if hardware supports BF16
    backend = jax.default_backend()
    
    if backend == 'gpu':
        # NVIDIA GPUs with Tensor Cores support BF16
        _USE_BFLOAT16 = True
        print("✓ BFloat16 mixed precision enabled (2x memory savings)")
        return True
    elif backend == 'tpu':
        # TPUs natively support BF16
        _USE_BFLOAT16 = True
        print("✓ BFloat16 mixed precision enabled (native TPU support)")
        return True
    else:
        # CPU or Metal - use FP32 for stability
        _USE_BFLOAT16 = False
        print("⚪ BFloat16 disabled (not supported on this backend)")
        return False


def is_bf16_enabled() -> bool:
    """Check if BF16 is currently enabled."""
    return _USE_BFLOAT16


def get_dtype():
    """Get the compute dtype based on precision setting."""
    return jnp.bfloat16 if _USE_BFLOAT16 else jnp.float32


def get_param_dtype():
    """
    Get the parameter storage dtype.
    
    We store params in FP32 for precision, but compute in BF16.
    """
    return jnp.float32  # Always store in FP32


def cast_to_compute(x):
    """Cast tensor to compute dtype (BF16 or FP32)."""
    if _USE_BFLOAT16:
        return x.astype(jnp.bfloat16)
    return x


def cast_to_output(x):
    """Cast tensor back to FP32 for output/loss computation."""
    if x.dtype == jnp.bfloat16:
        return x.astype(jnp.float32)
    return x


def mixed_precision_policy():
    """
    Get the mixed precision policy dictionary.
    
    Returns dict with:
    - compute_dtype: dtype for forward/backward passes
    - param_dtype: dtype for parameter storage
    - output_dtype: dtype for outputs
    """
    return {
        'compute_dtype': get_dtype(),
        'param_dtype': jnp.float32,
        'output_dtype': jnp.float32,
    }


class MixedPrecisionWrapper:
    """
    Wrapper for mixed precision forward pass.
    
    Usage:
        wrapper = MixedPrecisionWrapper()
        
        @wrapper
        def forward(params, x):
            # This runs in BF16 when enabled
            return model.apply(params, x)
    """
    
    def __call__(self, fn):
        """Wrap a function with mixed precision casts."""
        def wrapped(*args, **kwargs):
            # Cast inputs to compute dtype
            args = jax.tree.map(
                lambda x: cast_to_compute(x) if hasattr(x, 'dtype') else x,
                args
            )
            
            # Run forward pass
            outputs = fn(*args, **kwargs)
            
            # Cast outputs back to FP32
            outputs = jax.tree.map(
                lambda x: cast_to_output(x) if hasattr(x, 'dtype') else x,
                outputs
            )
            
            return outputs
        
        return wrapped


def apply_bf16_to_model(apply_fn):
    """
    Wrap a model's apply function with BF16 mixed precision.
    
    Args:
        apply_fn: Model's apply function (e.g., agent.apply)
        
    Returns:
        Wrapped apply function that uses BF16 for compute
    """
    if not _USE_BFLOAT16:
        return apply_fn
    
    def bf16_apply(params, x, *args, **kwargs):
        # Cast input to BF16
        x_bf16 = x.astype(jnp.bfloat16)
        
        # Run inference (Dense layers auto-cast)
        outputs = apply_fn(params, x_bf16, *args, **kwargs)
        
        # Cast outputs back to FP32
        outputs = jax.tree.map(
            lambda o: o.astype(jnp.float32) if hasattr(o, 'dtype') else o,
            outputs
        )
        
        return outputs
    
    return bf16_apply
