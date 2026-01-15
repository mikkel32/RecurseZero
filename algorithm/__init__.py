"""
RecurseZero Algorithm Package.

Provides:
- Muesli policy optimization
- Proper Value Equivalence (PVE)
- Confidence metrics
"""

from .muesli import (
    PolicyHead,
    muesli_policy_gradient_loss,
    compute_muesli_targets,
)
from .pve import (
    ValueHead,
    RewardHead,
    pve_loss,
    value_to_win_probability,
    win_probability_to_centipawn,
    ConfidenceMetrics,
)

__all__ = [
    'PolicyHead',
    'muesli_policy_gradient_loss',
    'compute_muesli_targets',
    'ValueHead',
    'RewardHead',
    'pve_loss',
    'value_to_win_probability',
    'win_probability_to_centipawn',
    'ConfidenceMetrics',
]
