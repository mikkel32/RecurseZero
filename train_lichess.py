#!/usr/bin/env python3
"""
Lichess Database Training for RecurseZero - MEMORY OPTIMIZED

Uses streaming parser with numpy arrays to avoid memory explosion.
All training 100% on GPU (zero CPU during training).

Usage:
    python train_lichess.py --games 50000
"""

import os
import sys
import time
import pickle
import argparse
import subprocess
import gc
from typing import List, Tuple
from functools import partial

print("=" * 60)
print("🎯 RecurseZero - Lichess Human Data Training")
print("=" * 60)
print()

# Set XLA flags BEFORE JAX import
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
import optax

print(f"✓ JAX loaded (backend: {jax.default_backend()})")

# Auto-install dependencies
def install_if_missing(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...", flush=True)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install_if_missing(['chess', 'zstandard', 'requests', 'tqdm'])

import chess
import chess.pgn
import zstandard
import requests
import io
import re
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# GPU MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                return {
                    'gpu': int(parts[0].strip()),
                    'mem_used': int(parts[1].strip()),
                    'mem_total': int(parts[2].strip()),
                    'temp': int(parts[3].strip()),
                }
    except:
        pass
    return None

def format_gpu_stats():
    stats = get_gpu_stats()
    if stats:
        return f"GPU:{stats['gpu']}% | {stats['mem_used']//1024}G/{stats['mem_total']//1024}G | {stats['temp']}°C"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
DATA_DIR = "lichess_data"

def download_with_progress(url: str, filepath: str, max_bytes: int = None) -> str:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    if max_bytes and total_size > max_bytes:
        print(f"   File is {total_size/(1024**3):.1f} GB, limiting to {max_bytes/(1024**3):.1f} GB")
        total_size = max_bytes
    
    downloaded = 0
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="   Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    if max_bytes and downloaded >= max_bytes:
                        break
    return filepath

def get_lichess_database(month: str = "2019-09", max_gb: float = 1.0) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"lichess_{month}.pgn.zst")
    
    if os.path.exists(filepath):
        print(f"✓ Found cached: {filepath}")
        return filepath
    
    url = LICHESS_DB_URL.format(month=month)
    print(f"📥 Downloading Lichess {month}")
    
    try:
        download_with_progress(url, filepath, int(max_gb * 1024**3))
        print(f"✓ Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT PGN PARSER
# Uses numpy arrays directly - NO Python lists of lists!
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_PATTERN = re.compile(r'\[%eval ([+-]?\d+\.?\d*|#[+-]?\d+)\]')

def extract_eval(comment: str) -> float:
    """Extract Stockfish eval, normalized to [-1, 1]."""
    if not comment:
        return None
    match = EVAL_PATTERN.search(comment)
    if not match:
        return None
    
    eval_str = match.group(1)
    if eval_str.startswith('#'):
        return 1.0 if int(eval_str[1:]) > 0 else -1.0
    
    cp = float(eval_str)
    # Sigmoid normalization
    return max(-1.0, min(1.0, 2.0 / (1.0 + 10.0 ** (-cp / 4.0)) - 1.0))


def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Convert board to numpy array (17 planes, 8x8).
    
    MEMORY EFFICIENT: Returns numpy array, not Python lists!
    """
    planes = np.zeros((17, 8, 8), dtype=np.float32)
    
    # Piece planes (0-11)
    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for piece_type in range(1, 7):
            plane_idx = color_idx * 6 + (piece_type - 1)
            for sq in board.pieces(piece_type, color):
                rank, file = divmod(sq, 8)
                planes[plane_idx, rank, file] = 1.0
    
    # Castling (12-15)
    planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Side to move (16)
    planes[16, :, :] = float(board.turn)
    
    return planes


def move_to_action(move: chess.Move) -> int:
    """Simple action encoding: from_sq * 64 + to_sq, clamped to 4672."""
    return (move.from_square * 64 + move.to_square) % 4672


def stream_positions(filepath: str, max_games: int, max_positions: int):
    """
    STREAMING parser - memory efficient!
    
    Yields (obs, action, target) as numpy arrays.
    """
    if not filepath or not os.path.exists(filepath):
        print("No file found, using sample data")
        for _ in range(min(1000, max_positions)):
            yield np.zeros((17, 8, 8), dtype=np.float32), 0, 0.0
        return
    
    dctx = zstandard.ZstdDecompressor()
    games_processed = 0
    positions_yielded = 0
    
    print(f"📖 Streaming from {filepath}...")
    
    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            with tqdm(total=max_positions, desc="   Processing", unit="pos") as pbar:
                while games_processed < max_games and positions_yielded < max_positions:
                    try:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            break
                        
                        # Get result
                        result_str = game.headers.get('Result', '*')
                        if result_str == '1-0':
                            white_result = 1.0
                        elif result_str == '0-1':
                            white_result = -1.0
                        else:
                            white_result = 0.0
                        
                        board = game.board()
                        
                        for node in game.mainline():
                            if positions_yielded >= max_positions:
                                break
                            
                            move = node.move
                            
                            # Get eval or use result
                            sf_eval = extract_eval(node.comment)
                            if sf_eval is not None:
                                target = sf_eval if board.turn == chess.WHITE else -sf_eval
                            else:
                                target = white_result if board.turn == chess.WHITE else -white_result
                            
                            # Board as numpy array (memory efficient!)
                            obs = board_to_array(board)
                            action = move_to_action(move)
                            
                            yield obs, action, target
                            positions_yielded += 1
                            pbar.update(1)
                            
                            board.push(move)
                        
                        games_processed += 1
                        
                    except Exception:
                        continue
    
    print(f"✓ Streamed {positions_yielded:,} positions from {games_processed:,} games")


def collect_positions(filepath: str, max_games: int, max_positions: int):
    """
    Collect positions into numpy arrays (memory efficient).
    
    Pre-allocates arrays to avoid memory fragmentation.
    """
    print(f"📊 Pre-allocating arrays for {max_positions:,} positions...")
    
    # Pre-allocate (this is key for memory efficiency!)
    obs_array = np.zeros((max_positions, 17, 8, 8), dtype=np.float32)
    actions_array = np.zeros(max_positions, dtype=np.int16)
    targets_array = np.zeros(max_positions, dtype=np.float32)
    
    idx = 0
    for obs, action, target in stream_positions(filepath, max_games, max_positions):
        obs_array[idx] = obs
        actions_array[idx] = action
        targets_array[idx] = target
        idx += 1
        
        if idx >= max_positions:
            break
    
    # Trim to actual size
    obs_array = obs_array[:idx]
    actions_array = actions_array[:idx]
    targets_array = targets_array[:idx]
    
    print(f"✓ Collected {idx:,} positions")
    print(f"   Memory: {obs_array.nbytes / (1024**3):.2f} GB")
    
    return obs_array, actions_array, targets_array


# ═══════════════════════════════════════════════════════════════════════════════
# GPU DATA LOADING - MEMORY OPTIMIZED
# Uses Int8 quantization to fit more data in VRAM
# ═══════════════════════════════════════════════════════════════════════════════

def load_to_gpu(obs_np, actions_np, targets_np):
    """
    Transfer numpy arrays to GPU with Int8 quantization.
    
    17 planes × Int8 = 4x memory savings vs 119 planes × FP32!
    """
    print("📤 Transferring to GPU VRAM...")
    
    # Transpose: (N, 17, 8, 8) -> (N, 8, 8, 17)
    obs_np = np.transpose(obs_np, (0, 2, 3, 1))
    
    N = obs_np.shape[0]
    
    # Use Int8 quantization (4x memory savings)
    # Piece planes are 0 or 1, so Int8 is perfect
    obs_int8 = (obs_np * 127).astype(np.int8)
    
    print(f"   Positions: {N:,}")
    print(f"   Obs memory: {obs_int8.nbytes / (1024**3):.2f} GB (Int8)")
    
    # Transfer to GPU
    obs = jax.device_put(jnp.array(obs_int8))
    actions = jax.device_put(jnp.array(actions_np.astype(np.int32)))
    targets = jax.device_put(jnp.array(targets_np))
    
    jax.block_until_ready(obs)
    
    total_bytes = obs.nbytes + actions.nbytes + targets.nbytes
    print(f"✓ Total VRAM: {total_bytes / (1024**3):.2f} GB")
    
    return obs, actions, targets


# ═══════════════════════════════════════════════════════════════════════════════
# GPU TRAINING
# Handles Int8 quantized 17-plane input, pads to 119 during forward pass
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=['batch_size'])
def sample_batch(key, obs, actions, targets, batch_size):
    n = obs.shape[0]
    indices = jax.random.choice(key, n, shape=(batch_size,), replace=False)
    return obs[indices], actions[indices], targets[indices]


@jax.jit
def train_step(state, obs, actions, targets):
    """
    Training step that handles Int8 quantized 17-plane input.
    Dequantizes and pads to 119 planes during forward pass only.
    """
    def loss_fn(params):
        # Dequantize Int8 -> FP32 and pad to 119 planes
        obs_float = obs.astype(jnp.float32) / 127.0
        obs_padded = jnp.pad(obs_float, ((0, 0), (0, 0), (0, 0), (0, 119 - 17)))
        
        policy_logits, values, _ = state.apply_fn(params, obs_padded)
        values = jnp.squeeze(values, -1)
        
        log_probs = jax.nn.log_softmax(policy_logits)
        one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
        policy_loss = -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))
        
        value_loss = jnp.mean((values - targets) ** 2)
        
        predicted = jnp.argmax(policy_logits, axis=-1)
        accuracy = jnp.mean(predicted == actions)
        
        return policy_loss + 0.5 * value_loss, {'policy_loss': policy_loss, 'value_loss': value_loss, 'accuracy': accuracy}
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, {'total_loss': loss, **aux}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', default='2019-09')
    # Conservative settings that fit in 22GB VRAM
    # Note: Forward pass pads 17→119 planes, so batch size affects VRAM a lot
    parser.add_argument('--games', type=int, default=50000, help='Max games')
    parser.add_argument('--positions', type=int, default=500000, help='Max positions')
    parser.add_argument('--steps', type=int, default=15000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (4096 is safe)')
    parser.add_argument('--max_gb', type=float, default=1.0, help='Max download size')
    parser.add_argument('--output', default='lichess_model.pkl')
    args = parser.parse_args()
    
    start_total = time.time()
    
    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Games: {args.games:,} | Positions: {args.positions:,}")
    print(f"  Steps: {args.steps:,} | Batch: {args.batch_size:,}")
    
    # Download
    print()
    print("=" * 60)
    print("DOWNLOAD")
    print("=" * 60)
    db_path = get_lichess_database(args.month, args.max_gb)
    
    # Collect positions (memory efficient)
    print()
    print("=" * 60)
    print("PARSING (Memory Efficient)")
    print("=" * 60)
    obs_np, actions_np, targets_np = collect_positions(db_path, args.games, args.positions)
    
    # Load to GPU
    print()
    print("=" * 60)
    print("GPU LOADING")
    print("=" * 60)
    obs, actions, targets = load_to_gpu(obs_np, actions_np, targets_np)
    
    # Free CPU memory
    del obs_np, actions_np, targets_np
    gc.collect()
    
    # Model
    print()
    print("=" * 60)
    print("MODEL INIT")
    print("=" * 60)
    
    from model.agent import RecurseZeroAgentSimple
    from flax.training import train_state
    
    agent = RecurseZeroAgentSimple(num_actions=4672)
    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, 8, 8, 119))
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy)
    
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model: {param_count:,} parameters")
    
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=1e-3))
    
    class TrainState(train_state.TrainState):
        pass
    
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=optimizer)
    
    # JIT warmup
    key, batch_key = jax.random.split(key)
    b_obs, b_act, b_tgt = sample_batch(batch_key, obs, actions, targets, args.batch_size)
    state, _ = train_step(state, b_obs, b_act, b_tgt)
    jax.block_until_ready(state.step)
    print("✓ JIT compiled")
    
    # Training
    print()
    print("=" * 60)
    print(f"TRAINING ({args.steps:,} steps) - 100% GPU")
    print("=" * 60)
    print(f"   {format_gpu_stats()}")
    print()
    
    train_start = time.time()
    
    with tqdm(total=args.steps, desc="Training", unit="step") as pbar:
        for step in range(args.steps):
            key, batch_key = jax.random.split(key)
            b_obs, b_act, b_tgt = sample_batch(batch_key, obs, actions, targets, args.batch_size)
            state, metrics = train_step(state, b_obs, b_act, b_tgt)
            
            if step % 100 == 0:
                jax.block_until_ready(metrics['total_loss'])
                elapsed = time.time() - train_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                gpu = format_gpu_stats()
                pbar.set_postfix_str(f"loss={float(metrics['total_loss']):.3f} | acc={float(metrics['accuracy']):.1%} | {speed:.0f}s/s | {gpu}")
            
            pbar.update(1)
    
    # Save
    print()
    print("=" * 60)
    print("💾 SAVING")
    print("=" * 60)
    
    from training.checkpoint import export_for_inference
    export_for_inference(state.params, args.output)
    
    total_time = time.time() - start_total
    print()
    print("=" * 60)
    print("✅ COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final accuracy: {float(metrics['accuracy']):.1%}")
    print(f"  Model: {args.output}")
    print()


if __name__ == "__main__":
    main()
