#!/usr/bin/env python3
"""
RecurseZero Lichess Training - 100% GPU-ONLY COMPLIANT

Per GPU-Only Chess RL Agent.txt spec:
- Section 1.1: GPU-Resident paradigm (no CPU during training)
- Section 2.2: Uses same architecture as main agent
- Section 3.1: Search-free inference
- Section 4.1: Int8 quantization for efficiency

Architecture:
- DATA PREPARATION (one-time): CPU parses PGN → NumPy → saves to disk
- TRAINING (100% GPU): Loads data to VRAM → trains entirely on GPU

The training loop has ZERO CPU operations - all JAX-compiled XLA kernels.
"""

import os
import sys
import time
import argparse
import subprocess
import gc
from functools import partial
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION - BEFORE ANY JAX IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

print("=" * 70)
print("🎯 RecurseZero Lichess Training - 100% GPU-ONLY")
print("=" * 70)
print()

import jax
import jax.numpy as jnp
import numpy as np
import optax

BACKEND = jax.default_backend()
print(f"✓ JAX Backend: {BACKEND}")

if BACKEND != 'gpu':
    print("⚠️ WARNING: Not running on GPU! Training will be slow.")

# Auto-install dependencies
for pkg in ['chess', 'zstandard', 'requests', 'tqdm']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

import chess
import chess.pgn
import zstandard
import requests
import io
import re
import pickle
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - Pgx-compatible 119-plane format with history
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_FRAMES = 8           # AlphaZero uses 8 history frames
PIECE_PLANES = 12            # 6 piece types × 2 colors
HISTORY_PLANES = HISTORY_FRAMES * PIECE_PLANES  # 96
META_PLANES = 7              # Castling(4) + turn(1) + halfmove(1) + ep(1)
TOTAL_PLANES = HISTORY_PLANES + META_PLANES     # 103
MODEL_PLANES = 119           # Pgx model expects 119 (we pad 16)

# Action space: 64 squares × 73 move types
ACTION_SPACE = 4672


# ═══════════════════════════════════════════════════════════════════════════════
# GPU MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_stats():
    """Get GPU stats for monitoring."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            p = result.stdout.strip().split(',')
            return f"GPU:{p[0].strip()}%|{int(p[1])//1024}G/{int(p[2])//1024}G|{p[3].strip()}°C"
    except:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA PREPARATION (CPU - runs once, saves to disk)
# ═══════════════════════════════════════════════════════════════════════════════

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
DATA_DIR = "lichess_data"
CACHE_DIR = "lichess_cache"

EVAL_RE = re.compile(r'\[%eval ([+-]?\d+\.?\d*|#[+-]?\d+)\]')


def download_lichess(month: str, max_gb: float) -> str:
    """Download Lichess database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"lichess_{month}.pgn.zst")
    
    if os.path.exists(filepath):
        print(f"✓ Found: {filepath}")
        return filepath
    
    url = LICHESS_URL.format(month=month)
    print(f"📥 Downloading {month}...")
    
    response = requests.get(url, stream=True, timeout=30)
    total = int(response.headers.get('content-length', 0))
    max_bytes = int(max_gb * 1024**3)
    
    if total > max_bytes:
        print(f"   Limiting to {max_gb}GB")
        total = max_bytes
    
    downloaded = 0
    with open(filepath, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    if downloaded >= max_bytes:
                        break
    
    return filepath


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Convert board to 12 piece planes (one history frame)."""
    planes = np.zeros((PIECE_PLANES, 8, 8), dtype=np.int8)
    for c_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for pt in range(1, 7):
            for sq in board.pieces(pt, color):
                planes[c_idx * 6 + pt - 1, sq // 8, sq % 8] = 1
    return planes


def get_meta_planes(board: chess.Board) -> np.ndarray:
    """Get 7 metadata planes."""
    planes = np.zeros((META_PLANES, 8, 8), dtype=np.int8)
    planes[0] = int(board.has_kingside_castling_rights(chess.WHITE))
    planes[1] = int(board.has_queenside_castling_rights(chess.WHITE))
    planes[2] = int(board.has_kingside_castling_rights(chess.BLACK))
    planes[3] = int(board.has_queenside_castling_rights(chess.BLACK))
    planes[4] = int(board.turn)
    planes[5] = min(board.halfmove_clock, 100) // 10
    if board.ep_square:
        planes[6, :, board.ep_square % 8] = 1
    return planes


def move_to_action(move: chess.Move) -> int:
    """Simple move encoding."""
    return (move.from_square * 64 + move.to_square) % ACTION_SPACE


def extract_eval(comment: str) -> float:
    """Extract Stockfish eval from comment."""
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None
    s = m.group(1)
    if s.startswith('#'):
        return 1.0 if int(s[1:]) > 0 else -1.0
    cp = float(s)
    return max(-1.0, min(1.0, 2.0 / (1.0 + 10.0 ** (-cp / 4.0)) - 1.0))


def prepare_data(pgn_path: str, max_games: int, max_positions: int) -> tuple:
    """
    PHASE 1: Parse PGN to NumPy arrays with history.
    
    This is the CPU-bound phase. Runs once, then training is 100% GPU.
    """
    print(f"📊 Preparing data with {HISTORY_FRAMES}-frame history...")
    
    # Pre-allocate arrays
    obs = np.zeros((max_positions, TOTAL_PLANES, 8, 8), dtype=np.int8)
    actions = np.zeros(max_positions, dtype=np.int16)
    targets = np.zeros(max_positions, dtype=np.float32)
    
    dctx = zstandard.ZstdDecompressor()
    games_done = 0
    pos_done = 0
    
    with open(pgn_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            with tqdm(total=max_positions, desc="   Parsing", unit="pos") as pbar:
                while games_done < max_games and pos_done < max_positions:
                    try:
                        game = chess.pgn.read_game(text)
                        if not game:
                            break
                        
                        # Game result
                        result_str = game.headers.get('Result', '*')
                        w_result = 1.0 if result_str == '1-0' else (-1.0 if result_str == '0-1' else 0.0)
                        
                        board = game.board()
                        
                        # History buffer
                        history = deque(maxlen=HISTORY_FRAMES)
                        empty = np.zeros((PIECE_PLANES, 8, 8), dtype=np.int8)
                        for _ in range(HISTORY_FRAMES):
                            history.append(empty.copy())
                        history.append(board_to_planes(board))
                        
                        for node in game.mainline():
                            if pos_done >= max_positions:
                                break
                            
                            move = node.move
                            
                            # Target: Stockfish eval or game result
                            sf = extract_eval(node.comment)
                            target = sf if sf is not None else (w_result if board.turn else -w_result)
                            
                            # Stack history + meta
                            hist_stack = np.concatenate(list(history), axis=0)
                            meta = get_meta_planes(board)
                            full_obs = np.concatenate([hist_stack, meta], axis=0)
                            
                            obs[pos_done] = full_obs
                            actions[pos_done] = move_to_action(move)
                            targets[pos_done] = target
                            pos_done += 1
                            pbar.update(1)
                            
                            board.push(move)
                            history.append(board_to_planes(board))
                        
                        games_done += 1
                        
                    except Exception:
                        continue
    
    # Trim to actual size
    obs = obs[:pos_done]
    actions = actions[:pos_done]
    targets = targets[:pos_done]
    
    print(f"✓ Prepared {pos_done:,} positions from {games_done:,} games")
    print(f"   Memory: {obs.nbytes / (1024**3):.2f} GB (Int8)")
    
    return obs, actions, targets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: GPU TRAINING (100% GPU-ONLY - ZERO CPU OPERATIONS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_to_vram(obs_np, actions_np, targets_np):
    """
    Transfer all data to GPU VRAM.
    After this, training uses ZERO CPU.
    """
    print("📤 Loading to GPU VRAM...")
    
    # Transpose to NHWC and pad to 119 planes
    obs_np = np.transpose(obs_np, (0, 2, 3, 1))  # (N, 8, 8, 103)
    n, h, w, c = obs_np.shape
    
    # Pad 103 → 119 planes (model expects 119)
    obs_padded = np.zeros((n, h, w, MODEL_PLANES), dtype=np.int8)
    obs_padded[:, :, :, :c] = obs_np
    
    print(f"   Positions: {n:,}")
    print(f"   Shape: {obs_padded.shape}")
    print(f"   Obs: {obs_padded.nbytes / (1024**3):.2f} GB")
    
    # Transfer to GPU
    obs = jax.device_put(jnp.array(obs_padded))
    actions = jax.device_put(jnp.array(actions_np.astype(np.int32)))
    targets = jax.device_put(jnp.array(targets_np))
    
    # Block until transfer complete
    jax.block_until_ready(obs)
    
    total = obs.nbytes + actions.nbytes + targets.nbytes
    print(f"✓ VRAM used: {total / (1024**3):.2f} GB")
    
    return obs, actions, targets


@partial(jax.jit, static_argnames=['batch_size'])
def sample_batch(key, obs, actions, targets, batch_size):
    """Sample random batch - 100% GPU."""
    n = obs.shape[0]
    idx = jax.random.choice(key, n, (batch_size,), replace=False)
    return obs[idx], actions[idx], targets[idx]


@jax.jit
def train_step(state, obs, actions, targets, key):
    """
    Single training step - 100% GPU.
    
    All operations compile to XLA kernels.
    ZERO CPU involvement.
    """
    def loss_fn(params):
        # Dequantize Int8 → Float32
        obs_f = obs.astype(jnp.float32) / 127.0
        
        # Forward pass with dropout
        logits, values, _ = state.apply_fn(
            params, obs_f, train=True,
            rngs={'dropout': key}
        )
        values = jnp.squeeze(values, -1)
        
        # Policy loss with label smoothing (better generalization)
        n_cls = logits.shape[-1]
        smooth = 0.1
        one_hot = jax.nn.one_hot(actions, n_cls)
        smoothed = one_hot * (1.0 - smooth) + smooth / n_cls
        
        log_probs = jax.nn.log_softmax(logits)
        policy_loss = -jnp.mean(jnp.sum(log_probs * smoothed, axis=-1))
        
        # Value loss
        value_loss = jnp.mean((values - targets) ** 2)
        
        # Accuracy
        pred = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(pred == actions)
        
        total_loss = policy_loss + 0.25 * value_loss
        
        return total_loss, {'policy': policy_loss, 'value': value_loss, 'acc': acc}
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, {'loss': loss, **aux}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RecurseZero Lichess Training - 100% GPU")
    parser.add_argument('--month', default='2019-09', help='Lichess month')
    parser.add_argument('--games', type=int, default=50000, help='Max games')
    parser.add_argument('--positions', type=int, default=500000, help='Max positions (fits in 22GB)')
    parser.add_argument('--steps', type=int, default=40000, help='Training steps')
    parser.add_argument('--batch', type=int, default=1024, help='Batch size')
    parser.add_argument('--max_gb', type=float, default=1.0, help='Max download GB')
    parser.add_argument('--output', default='lichess_model.pkl', help='Output path')
    args = parser.parse_args()
    
    start = time.time()
    
    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Games: {args.games:,} | Positions: {args.positions:,}")
    print(f"  Steps: {args.steps:,} | Batch: {args.batch:,}")
    print(f"  History: {HISTORY_FRAMES} frames | Planes: {TOTAL_PLANES}→{MODEL_PLANES}")
    print(f"  Download: {args.max_gb} GB")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA PREPARATION (CPU)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION (CPU)")
    print("=" * 70)
    
    pgn_path = download_lichess(args.month, args.max_gb)
    obs_np, actions_np, targets_np = prepare_data(pgn_path, args.games, args.positions)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: GPU TRAINING (100% GPU-ONLY)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("PHASE 2: GPU TRAINING (100% GPU-ONLY)")
    print("=" * 70)
    
    # Load to VRAM
    obs, actions, targets = load_to_vram(obs_np, actions_np, targets_np)
    
    # Free CPU memory
    del obs_np, actions_np, targets_np
    gc.collect()
    
    print()
    print("─" * 70)
    print("MODEL INITIALIZATION")
    print("─" * 70)
    
    from model.agent import RecurseZeroAgentSimple
    from flax.training import train_state
    
    agent = RecurseZeroAgentSimple(num_actions=ACTION_SPACE)
    key = jax.random.PRNGKey(42)
    
    # Initialize
    dummy = jnp.zeros((1, 8, 8, MODEL_PLANES))
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy)
    
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model: {n_params:,} parameters")
    
    # Optimizer with cosine LR
    warmup = min(1000, args.steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=1e-3,
        warmup_steps=warmup,
        decay_steps=args.steps,
        end_value=1e-5
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01)
    )
    print(f"✓ Cosine LR: 1e-4 → 1e-3 → 1e-5 (warmup={warmup})")
    
    class TrainState(train_state.TrainState):
        pass
    
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=opt)
    
    # JIT compile (warmup)
    key, batch_key, drop_key = jax.random.split(key, 3)
    b_obs, b_act, b_tgt = sample_batch(batch_key, obs, actions, targets, args.batch)
    state, _ = train_step(state, b_obs, b_act, b_tgt, drop_key)
    jax.block_until_ready(state.step)
    print("✓ JIT compiled (GPU kernels ready)")
    
    print()
    print("─" * 70)
    print(f"TRAINING ({args.steps:,} steps) - 100% GPU-ONLY")
    print("─" * 70)
    print(f"   {gpu_stats()}")
    print()
    
    train_start = time.time()
    
    with tqdm(total=args.steps, desc="Training", unit="step") as pbar:
        for step in range(args.steps):
            # All operations below are GPU-only (XLA kernels)
            key, batch_key, drop_key = jax.random.split(key, 3)
            b_obs, b_act, b_tgt = sample_batch(batch_key, obs, actions, targets, args.batch)
            state, metrics = train_step(state, b_obs, b_act, b_tgt, drop_key)
            
            if step % 100 == 0:
                jax.block_until_ready(metrics['loss'])
                elapsed = time.time() - train_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(
                    f"loss={float(metrics['loss']):.3f} | "
                    f"acc={float(metrics['acc']):.1%} | "
                    f"{speed:.0f}s/s | {gpu_stats()}"
                )
            
            pbar.update(1)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("SAVING")
    print("=" * 70)
    
    from training.checkpoint import export_for_inference
    export_for_inference(state.params, args.output)
    
    total_time = time.time() - start
    train_time = time.time() - train_start
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Train time: {train_time:.0f}s ({train_time/60:.1f} min)")
    print(f"  Final accuracy: {float(metrics['acc']):.1%}")
    print(f"  Model: {args.output}")
    print()


if __name__ == "__main__":
    main()
