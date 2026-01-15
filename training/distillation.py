"""
Distillation Training for RecurseZero.

Per GPU ONLY Chess RL Agent.txt spec section 4.2:
- Teacher: Larger DEQ-Transformer with stricter tolerance
- Student: Deployed agent optimized for 8GB VRAM
- Loss: KL divergence between teacher and student policies

This enables knowledge transfer from a stronger model to a faster one.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from functools import partial
from typing import Optional, Callable


class DistillationConfig:
    """Configuration for distillation training."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,  # Weight for distillation loss vs self-play loss
        teacher_checkpoint_path: Optional[str] = None
    ):
        """
        Args:
            temperature: Softmax temperature for softer probability distributions.
                         Higher values produce softer distributions with more
                         information about relative move quality.
            alpha: Balance between distillation and self-play.
                   alpha=1.0 means pure distillation
                   alpha=0.0 means pure self-play
            teacher_checkpoint_path: Path to teacher model checkpoint
        """
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_checkpoint_path = teacher_checkpoint_path


def kl_divergence(teacher_logits: jnp.ndarray, 
                  student_logits: jnp.ndarray,
                  legal_mask: jnp.ndarray,
                  temperature: float = 2.0) -> jnp.ndarray:
    """
    Computes KL divergence between teacher and student policies.
    
    D_KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
    
    Per spec 4.2:
    "The Student is trained to minimize the KL Divergence between its policy
    output and the Teacher's policy output."
    
    Args:
        teacher_logits: Raw logits from teacher model (B, A)
        student_logits: Raw logits from student model (B, A)
        legal_mask: Boolean mask for legal moves (B, A)
        temperature: Softmax temperature for softer distributions
        
    Returns:
        kl_loss: Scalar KL divergence loss
    """
    # Mask illegal moves
    teacher_masked = jnp.where(legal_mask, teacher_logits, -1e9)
    student_masked = jnp.where(legal_mask, student_logits, -1e9)
    
    # Apply temperature scaling (softer distributions = more knowledge transfer)
    teacher_scaled = teacher_masked / temperature
    student_scaled = student_masked / temperature
    
    # Compute probabilities
    teacher_probs = jax.nn.softmax(teacher_scaled, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_scaled, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_scaled, axis=-1)
    
    # KL divergence: sum(P * (log P - log Q))
    kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    
    # Scale by temperature^2 (standard practice in distillation)
    kl = kl * (temperature ** 2)
    
    return jnp.mean(kl)


def distillation_loss(
    teacher_logits: jnp.ndarray,
    teacher_value: jnp.ndarray,
    student_logits: jnp.ndarray,
    student_value: jnp.ndarray,
    legal_mask: jnp.ndarray,
    temperature: float = 2.0
) -> tuple[jnp.ndarray, dict]:
    """
    Complete distillation loss combining policy and value distillation.
    
    Args:
        teacher_logits: Teacher policy logits (B, A)
        teacher_value: Teacher value prediction (B, 1)
        student_logits: Student policy logits (B, A)
        student_value: Student value prediction (B, 1)
        legal_mask: Legal action mask (B, A)
        temperature: Softmax temperature
        
    Returns:
        total_loss: Combined distillation loss
        metrics: Dictionary of component losses
    """
    # Policy distillation (KL divergence)
    policy_loss = kl_divergence(
        teacher_logits, student_logits, legal_mask, temperature
    )
    
    # Value distillation (MSE)
    value_loss = jnp.mean(jnp.square(
        jnp.squeeze(student_value, -1) - jnp.squeeze(teacher_value, -1)
    ))
    
    # Combined loss
    total_loss = policy_loss + 0.5 * value_loss
    
    metrics = {
        'distill_policy_loss': policy_loss,
        'distill_value_loss': value_loss,
        'distill_total_loss': total_loss,
    }
    
    return total_loss, metrics


@partial(jax.jit, static_argnames=('student_apply_fn', 'teacher_apply_fn'))
def distillation_train_step(
    student_state: train_state.TrainState,
    teacher_params: dict,
    observations: jnp.ndarray,
    legal_masks: jnp.ndarray,
    key: jax.Array,
    student_apply_fn: Callable,
    teacher_apply_fn: Callable,
    temperature: float = 2.0
):
    """
    Single distillation training step.
    
    Args:
        student_state: Training state for student model
        teacher_params: Frozen teacher model parameters
        observations: Batch of board observations
        legal_masks: Legal action masks
        key: PRNG key
        student_apply_fn: Student model forward function
        teacher_apply_fn: Teacher model forward function
        temperature: Distillation temperature
        
    Returns:
        new_state: Updated student training state
        metrics: Training metrics
    """
    # Teacher inference (no gradients)
    teacher_logits, teacher_values, _ = teacher_apply_fn(
        teacher_params, observations
    )
    
    def loss_fn(params):
        student_logits, student_values, _ = student_apply_fn(params, observations)
        loss, metrics = distillation_loss(
            teacher_logits, teacher_values,
            student_logits, student_values,
            legal_masks, temperature
        )
        return loss, metrics
    
    grads, metrics = jax.grad(loss_fn, has_aux=True)(student_state.params)
    
    # Gradient clipping
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    new_state = student_state.apply_gradients(grads=grads)
    
    return new_state, metrics


class DistillationTrainer:
    """
    High-level trainer for knowledge distillation.
    
    Usage:
        trainer = DistillationTrainer(
            teacher_model=large_agent,
            student_model=fast_agent,
            config=DistillationConfig(temperature=2.0, alpha=0.7)
        )
        trainer.load_teacher("teacher_checkpoint.pkl")
        trainer.distill(env, num_steps=10000)
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.teacher_params = None
        
    def load_teacher(self, checkpoint_path: str):
        """Load teacher model parameters from checkpoint."""
        import pickle
        with open(checkpoint_path, 'rb') as f:
            self.teacher_params = pickle.load(f)
        print(f"✓ Loaded teacher from {checkpoint_path}")
        
    def save_student(self, checkpoint_path: str, student_params: dict):
        """Save student model parameters to checkpoint."""
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(student_params, f)
        print(f"✓ Saved student to {checkpoint_path}")
