import jax
import os

def setup_jax_platform():
    """
    Configures JAX for the available hardware (Metal/MPS or CUDA).
    Enforces precision policies for RecurseZero.
    """
    # Text Section 4.3: "Implementations of JAX on Metal... do not support 64-bit efficiently".
    # "RecurseZero adapts to this by enforcing strict FP32".
    
    # Check if Metal is available (roughly)
    # JAX usually auto-detects, but we can configure flags.
    
    # Disable FP64 to prevent accidental usage on MPS which might fallback to CPU or error
    jax.config.update("jax_enable_x64", False)
    
    # Default matmul precision
    # "Tensor Core usage" on NVIDIA requires correct precision settings.
    # On MPS, we stick to default or 'tensorfloat32' equivalent logic if available.
    if jax.default_backend() == 'gpu':
        # Default to highest efficiency
        pass 
        
    print(f"RecurseZero Hardware Setup: Backend={jax.default_backend()}, X64=False")

def ensure_fp32(x):
    """
    Casts x to float32 if it is float64.
    """
    if x.dtype == jax.numpy.float64:
        return x.astype(jax.numpy.float32)
    return x
