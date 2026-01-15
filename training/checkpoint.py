"""
Checkpoint utilities for RecurseZero model saving and loading.
"""

import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


def save_checkpoint(
    params: Any,
    step: int,
    path: str = "checkpoints",
    prefix: str = "recurse_zero"
) -> str:
    """
    Save model checkpoint.
    
    Args:
        params: Model parameters (pytree)
        step: Current training step
        path: Directory to save checkpoints
        prefix: Filename prefix
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'params': jax.device_get(params),  # Move to CPU for saving
        'step': step,
    }
    
    filepath = os.path.join(path, f"{prefix}_step_{step}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Also save as "latest"
    latest_path = os.path.join(path, f"{prefix}_latest.pkl")
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    return filepath


def load_checkpoint(
    path: str,
    prefix: str = "recurse_zero",
    step: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        path: Directory containing checkpoints
        prefix: Filename prefix
        step: Specific step to load (None = latest)
        
    Returns:
        Checkpoint dict with 'params' and 'step', or None if not found
    """
    if step is not None:
        filepath = os.path.join(path, f"{prefix}_step_{step}.pkl")
    else:
        filepath = os.path.join(path, f"{prefix}_latest.pkl")
    
    if not os.path.exists(filepath):
        print(f"⚠ Checkpoint not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint


def get_param_count(params: Any) -> int:
    """Count total parameters in model."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def export_for_inference(
    params: Any,
    output_path: str = "model_export.pkl"
) -> str:
    """
    Export model for inference (smaller file, no optimizer state).
    
    Args:
        params: Model parameters
        output_path: Output file path
        
    Returns:
        Path to exported model
    """
    # Convert to numpy for portability
    params_np = jax.device_get(params)
    
    export = {
        'params': params_np,
        'param_count': get_param_count(params),
        'version': '1.0',
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(export, f)
    
    print(f"✓ Model exported to {output_path}")
    print(f"  Parameters: {export['param_count']:,}")
    
    return output_path
