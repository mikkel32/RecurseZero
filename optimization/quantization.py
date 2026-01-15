"""
Int8 Quantization Module for RecurseZero.

Uses JAX AQT (Accurate Quantized Training) to enable Int8 inference
for 2-4x speedup on Tensor Cores.

Falls back gracefully if AQT is not available.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any

# Global config flag
_USE_QUANTIZATION = False
_AQT_CONFIG = None

def init_quantization(enable: bool = True):
    """
    Initialize AQT quantization.
    
    Args:
        enable: Whether to enable Int8 quantization
        
    Returns:
        True if quantization was enabled successfully
    """
    global _USE_QUANTIZATION, _AQT_CONFIG
    
    if not enable:
        _USE_QUANTIZATION = False
        _AQT_CONFIG = None
        print("⚪ Quantization disabled")
        return False
    
    try:
        import aqt.jax.v2.flax.aqt_flax as aqt
        import aqt.jax.v2.config as aqt_config
        
        # Create Int8 config for forward and backward passes
        _AQT_CONFIG = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
        _USE_QUANTIZATION = True
        print("✓ Int8 Quantization enabled (AQT)")
        return True
        
    except ImportError as e:
        print(f"⚠ AQT not available, running in FP32 mode: {e}")
        _USE_QUANTIZATION = False
        _AQT_CONFIG = None
        return False

def get_quantized_dot_general():
    """
    Get a quantized dot_general function if available.
    
    Returns:
        AqtDotGeneral instance or None
    """
    global _USE_QUANTIZATION, _AQT_CONFIG
    
    if not _USE_QUANTIZATION or _AQT_CONFIG is None:
        return None
    
    try:
        import aqt.jax.v2.flax.aqt_flax as aqt
        return aqt.AqtDotGeneral(_AQT_CONFIG)
    except Exception:
        return None

def is_quantization_enabled() -> bool:
    """Check if quantization is currently enabled."""
    return _USE_QUANTIZATION


class QuantizedDense(nn.Module):
    """
    Dense layer with optional Int8 quantization.
    
    Automatically uses AQT if enabled, otherwise standard Dense.
    """
    features: int
    use_bias: bool = True
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x):
        dot_general = get_quantized_dot_general()
        
        if dot_general is not None:
            # Use quantized matmul
            return nn.Dense(
                features=self.features,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dot_general=dot_general
            )(x)
        else:
            # Fallback to standard Dense
            return nn.Dense(
                features=self.features,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)


class QuantizedMLP(nn.Module):
    """
    MLP block with optional Int8 quantization.
    """
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = QuantizedDense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = QuantizedDense(self.output_dim)(x)
        return x
