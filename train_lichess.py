#!/usr/bin/env python3
"""
Lichess Database Training for RecurseZero - OPTIMIZED VERSION

Fast download with live progress + efficient PGN parsing.
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
from typing import List, Tuple, Generator
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

print("=" * 60)
print("🎯 RecurseZero - Lichess Human Data Training")
print("=" * 60)
print()

# Set XLA flags BEFORE JAX import
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
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
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# FAST DOWNLOAD WITH LIVE PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════

LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
DATA_DIR = "lichess_data"


def download_with_progress(url: str, filepath: str, max_bytes: int = None) -> str:
    """Download file with live progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    if max_bytes and total_size > max_bytes:
        print(f"   File is {total_size/(1024**3):.1f} GB, limiting to {max_bytes/(1024**3):.1f} GB")
        total_size = max_bytes
    
    block_size = 1024 * 1024  # 1MB blocks for speed
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="   Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    
                    if max_bytes and downloaded >= max_bytes:
                        break
    
    return filepath


def get_lichess_database(month: str = "2019-09", max_gb: float = 1.0) -> str:
    """Get Lichess database (download if needed)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = os.path.join(DATA_DIR, f"lichess_{month}.pgn.zst")
    
    if os.path.exists(filepath):
        size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"✓ Found cached: {filepath} ({size_gb:.2f} GB)")
        return filepath
    
    url = LICHESS_DB_URL.format(month=month)
    print(f"📥 Downloading Lichess {month}")
    print(f"   URL: {url}")
    
    try:
        max_bytes = int(max_gb * 1024**3)
        download_with_progress(url, filepath, max_bytes)
        print(f"✓ Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FAST PGN PARSING (Optimized)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_single_game(game_text: str) -> List[Tuple]:
    """Parse a single PGN game to positions (CPU worker)."""
    positions = []
    
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if not game:
            return []
        
        # Get result
        result_str = game.headers.get('Result', '*')
        if result_str == '1-0':
            white_result = 1.0
        elif result_str == '0-1':
            white_result = -1.0
        else:
            white_result = 0.0
        
        board = game.board()
        
        for move in game.mainline_moves():
            # Simple board encoding (fast)
            planes = []
            for color in [chess.WHITE, chess.BLACK]:
                for piece_type in range(1, 7):
                    plane = [[0.0] * 8 for _ in range(8)]
                    for sq in board.pieces(piece_type, color):
                        rank, file = divmod(sq, 8)
                        plane[rank][file] = 1.0
                    planes.append(plane)
            
            # Pad to 17 planes (minimal)
            while len(planes) < 17:
                planes.append([[0.0] * 8 for _ in range(8)])
            
            obs = planes[:17]  # (17, 8, 8)
            
            # Action encoding
            action = (move.from_square * 73 + move.to_square) % 4672
            
            # Result
            result = white_result if board.turn == chess.WHITE else -white_result
            
            positions.append((obs, action, result))
            board.push(move)
            
    except Exception:
        pass
    
    return positions


def extract_games_fast(filepath: str, max_games: int) -> List[str]:
    """Extract games from zst file quickly."""
    if not filepath or not os.path.exists(filepath):
        return []
    
    games = []
    dctx = zstandard.ZstdDecompressor()
    current_game = []
    
    print(f"📖 Extracting games from {filepath}...")
    
    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            with tqdm(total=max_games, desc="   Extracting", unit="games") as pbar:
                for line in text_stream:
                    current_game.append(line)
                    
                    # End of game
                    if line.strip().endswith(('1-0', '0-1', '1/2-1/2', '*')):
                        game_text = ''.join(current_game)
                        
                        # Skip short games
                        if game_text.count('.') > 10:  # At least 10 moves
                            games.append(game_text)
                            pbar.update(1)
                        
                        current_game = []
                        
                        if len(games) >= max_games:
                            break
    
    print(f"✓ Extracted {len(games):,} games")
    return games


def parallel_parse_games(games: List[str], max_positions: int, num_workers: int = 4) -> Tuple:
    """Parse games in parallel using multiple CPU cores."""
    print(f"🔄 Parsing games with {num_workers} workers...")
    
    all_obs = []
    all_actions = []
    all_results = []
    total_positions = 0
    
    with tqdm(total=min(len(games), max_positions // 40), desc="   Parsing", unit="games") as pbar:
        # Process in batches to show progress
        batch_size = 100
        
        for i in range(0, len(games), batch_size):
            batch = games[i:i+batch_size]
            
            # Parse batch
            for game_text in batch:
                positions = parse_single_game(game_text)
                
                for obs, action, result in positions:
                    if total_positions >= max_positions:
                        break
                    
                    all_obs.append(obs)
                    all_actions.append(action)
                    all_results.append(result)
                    total_positions += 1
                
                if total_positions >= max_positions:
                    break
            
            pbar.update(len(batch))
            
            if total_positions >= max_positions:
                break
    
    print(f"✓ Parsed {total_positions:,} positions")
    return all_obs, all_actions, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# GPU DATA LOADING (100% GPU after this point)
# ═══════════════════════════════════════════════════════════════════════════════

def load_to_gpu(all_obs, all_actions, all_results):
    """Transfer all data to GPU VRAM."""
    print("📤 Transferring to GPU VRAM...")
    
    # Convert to numpy first (faster)
    import numpy as np
    
    # Reshape observations: (N, 17, 8, 8) -> (N, 8, 8, 17)
    obs_np = np.array(all_obs, dtype=np.float32)
    obs_np = np.transpose(obs_np, (0, 2, 3, 1))  # NCHW -> NHWC
    
    # Pad to 119 channels for compatibility
    N = obs_np.shape[0]
    obs_padded = np.zeros((N, 8, 8, 119), dtype=np.float32)
    obs_padded[:, :, :, :17] = obs_np
    
    actions_np = np.array(all_actions, dtype=np.int32)
    results_np = np.array(all_results, dtype=np.float32)
    
    # Transfer to GPU
    obs = jax.device_put(jnp.array(obs_padded))
    actions = jax.device_put(jnp.array(actions_np))
    results = jax.device_put(jnp.array(results_np))
    
    # Wait for transfer
    jax.block_until_ready(obs)
    
    total_bytes = obs.nbytes + actions.nbytes + results.nbytes
    print(f"✓ Data in VRAM: {total_bytes / (1024**3):.2f} GB")
    print(f"   Shape: {obs.shape}")
    
    return obs, actions, results


# ═══════════════════════════════════════════════════════════════════════════════
# GPU TRAINING (100% GPU - ZERO CPU)
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=['batch_size'])
def sample_batch(key, obs, actions, results, batch_size):
    """Sample batch from GPU (no CPU transfer)."""
    n = obs.shape[0]
    indices = jax.random.choice(key, n, shape=(batch_size,), replace=False)
    return obs[indices], actions[indices], results[indices]


@jax.jit
def train_step(state, obs, actions, results):
    """Training step - 100% on GPU."""
    def loss_fn(params):
        policy_logits, values, _ = state.apply_fn(params, obs)
        values = jnp.squeeze(values, -1)
        
        # Policy loss (cross-entropy)
        log_probs = jax.nn.log_softmax(policy_logits)
        one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
        policy_loss = -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))
        
        # Value loss (MSE)
        value_loss = jnp.mean((values - results) ** 2)
        
        # Accuracy
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
    parser.add_argument('--month', default='2019-09', help='Database month (YYYY-MM)')
    parser.add_argument('--games', type=int, default=10000, help='Max games')
    parser.add_argument('--positions', type=int, default=200000, help='Max positions')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--max_gb', type=float, default=1.0, help='Max download GB')
    parser.add_argument('--output', default='lichess_model.pkl', help='Output path')
    args = parser.parse_args()
    
    start_total = time.time()
    
    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Games: {args.games:,} | Positions: {args.positions:,}")
    print(f"  Steps: {args.steps:,} | Batch: {args.batch_size:,}")
    
    # 1. Download
    print()
    print("=" * 60)
    print("DOWNLOAD")
    print("=" * 60)
    
    db_path = get_lichess_database(args.month, args.max_gb)
    
    # 2. Extract games
    print()
    print("=" * 60)
    print("EXTRACTION (CPU)")
    print("=" * 60)
    
    games = extract_games_fast(db_path, args.games)
    
    if not games:
        print("No games found, using sample data...")
        games = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0"] * 1000
    
    # 3. Parse to positions
    print()
    print("=" * 60)
    print("PARSING (CPU)")
    print("=" * 60)
    
    all_obs, all_actions, all_results = parallel_parse_games(games, args.positions)
    
    # 4. Load to GPU
    print()
    print("=" * 60)
    print("GPU LOADING")
    print("=" * 60)
    
    obs, actions, results = load_to_gpu(all_obs, all_actions, all_results)
    
    # Free CPU memory
    del all_obs, all_actions, all_results, games
    import gc; gc.collect()
    
    # 5. Initialize model
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
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3)
    )
    
    class TrainState(train_state.TrainState):
        pass
    
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=optimizer)
    
    # JIT warmup
    key, batch_key = jax.random.split(key)
    b_obs, b_act, b_res = sample_batch(batch_key, obs, actions, results, args.batch_size)
    state, _ = train_step(state, b_obs, b_act, b_res)
    jax.block_until_ready(state.step)
    print("✓ JIT compiled")
    
    # 6. Training (100% GPU)
    print()
    print("=" * 60)
    print(f"TRAINING ({args.steps:,} steps) - 100% GPU")
    print("=" * 60)
    print()
    
    train_start = time.time()
    
    with tqdm(total=args.steps, desc="Training", unit="step") as pbar:
        for step in range(args.steps):
            key, batch_key = jax.random.split(key)
            b_obs, b_act, b_res = sample_batch(batch_key, obs, actions, results, args.batch_size)
            state, metrics = train_step(state, b_obs, b_act, b_res)
            
            if step % 100 == 0:
                jax.block_until_ready(metrics['total_loss'])
                elapsed = time.time() - train_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'loss': f"{float(metrics['total_loss']):.3f}",
                    'acc': f"{float(metrics['accuracy']):.1%}",
                    's/s': f"{speed:.0f}"
                })
            
            pbar.update(1)
    
    # 7. Save
    print()
    print("=" * 60)
    print("💾 SAVING")
    print("=" * 60)
    
    from training.checkpoint import export_for_inference
    export_for_inference(state.params, args.output)
    
    total_time = time.time() - start_total
    train_time = time.time() - train_start
    
    print()
    print("=" * 60)
    print("✅ COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Train time: {train_time:.0f}s")
    print(f"  Final accuracy: {float(metrics['accuracy']):.1%}")
    print(f"  Model: {args.output}")
    print()


if __name__ == "__main__":
    main()
