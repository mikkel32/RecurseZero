import jax
import jax.numpy as jnp
from jax import lax
import pgx._src.games.chess as chess_module

# Metal-safe XOR reduction to replace lax.reduce(..., bitwise_xor)
def safe_xor_reduce(x, axis=0):
    """
    Computes bitwise XOR reduction along axis 0 using a loop.
    This avoids the 'mhlo.reduce' with bitwise XOR on Metal which can crash.
    """
    # x shape assumed (N, M...) or (N,)
    # We reduce along dimension 0.
    
    def body_fn(i, val):
        return val ^ x[i]
    
    # Initialize with 0 (Identity for XOR)
    init_val = jnp.zeros(x.shape[1:], dtype=x.dtype)
    return lax.fori_loop(0, x.shape[0], body_fn, init_val)

def _zobrist_hash_metal_safe(state):
    """
    Monkeypatched version of _zobrist_hash from pgx.chess.
    Uses safe_xor_reduce instead of lax.reduce.
    """
    # Access consants from the module
    ZOBRIST_SIDE = chess_module.ZOBRIST_SIDE
    ZOBRIST_BOARD = chess_module.ZOBRIST_BOARD
    ZOBRIST_CASTLING = chess_module.ZOBRIST_CASTLING
    ZOBRIST_EN_PASSANT = chess_module.ZOBRIST_EN_PASSANT
    
    hash_ = lax.select(state.color == 0, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    
    # Board Hash
    to_reduce = ZOBRIST_BOARD[jnp.arange(64), state.board + 6]  # 0, ..., 12
    hash_ ^= safe_xor_reduce(to_reduce,axis=0)
    
    # Castling Hash
    to_reduce = jnp.where(state.castling_rights.reshape(-1, 1), ZOBRIST_CASTLING, 0)
    hash_ ^= safe_xor_reduce(to_reduce, axis=0)
    
    # En Passant Hash
    hash_ ^= ZOBRIST_EN_PASSANT[state.en_passant]
    
    return hash_

def apply_patch():
    """
    Applies the monkeypatch to pgx._src.games.chess._zobrist_hash
    """
    print("Applying Pygmalion Metal Patch to Pgx Chess...")
    chess_module._zobrist_hash = _zobrist_hash_metal_safe
    print("Patch applied successfully.")
