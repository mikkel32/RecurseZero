"""
RecurseZero Training Package.

Provides:
- GPU-resident training loop (Muesli + PVE)
- Knowledge distillation
- Training utilities
"""

from .loop import resident_train_step, TrainState, create_train_state
from .distillation import (
    DistillationConfig,
    DistillationTrainer,
    distillation_loss,
    kl_divergence,
    distillation_train_step,
)

__all__ = [
    'resident_train_step',
    'TrainState',
    'create_train_state',
    'DistillationConfig',
    'DistillationTrainer',
    'distillation_loss',
    'kl_divergence',
    'distillation_train_step',
]
