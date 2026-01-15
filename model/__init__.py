"""
RecurseZero Model Package.

GPU-Resident Chess Agent with:
- Deep Equilibrium Models (DEQ)
- Anderson Acceleration
- GTrXL Stabilization
- Chess-aware Positional Encodings
- Int8 Quantization (AQT)
"""

from .agent import RecurseZeroAgent, RecurseZeroAgentFast
from .deq import DeepEquilibriumModel
from .universal_transformer import UniversalTransformerBlock, ChessformerAttention
from .gtrxl import GTrXLGating
from .embeddings import ChessRelativePositionBias

__all__ = [
    'RecurseZeroAgent',
    'RecurseZeroAgentFast',
    'DeepEquilibriumModel',
    'UniversalTransformerBlock',
    'ChessformerAttention',
    'GTrXLGating',
    'ChessRelativePositionBias',
]
