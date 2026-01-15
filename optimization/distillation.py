import jax
import jax.numpy as jnp
import flax.linen as nn

def distillation_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    Computes KL Divergence loss for distillation.
    
    L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
    """
    teacher_probs = nn.softmax(teacher_logits / temperature)
    student_log_probs = nn.log_softmax(student_logits / temperature)
    
    # KL = Sum p_t * (log p_t - log p_s)
    # Relative entropy part: - Sum p_t * log p_s
    
    loss = -jnp.mean(jnp.sum(teacher_probs * student_log_probs, axis=-1))
    
    return loss * (temperature ** 2)
