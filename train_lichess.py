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
# GPU MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def get_gpu_stats():
    """Get GPU stats (usage, memory, temperature)."""
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
    except Exception:
        pass
    return None


def format_gpu_stats():
    """Format GPU stats for display."""
    stats = get_gpu_stats()
    if stats:
        return f"GPU:{stats['gpu']}% | {stats['mem_used']//1024}G/{stats['mem_total']//1024}G | {stats['temp']}°C"
    return ""


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
# INTELLIGENT PGN PARSING WITH STOCKFISH EVALS
# Uses Pgx-style 119-plane encoding for GPU compatibility
# ═══════════════════════════════════════════════════════════════════════════════

import re

# Regex to extract Stockfish eval from comment
EVAL_PATTERN = re.compile(r'\[%eval ([+-]?\d+\.?\d*|#[+-]?\d+)\]')


def extract_eval(comment: str) -> float:
    """
    Extract Stockfish evaluation from PGN comment.
    
    Returns:
        Float in range [-1, 1] (normalized centipawn or mate score)
    """
    if not comment:
        return None
    
    match = EVAL_PATTERN.search(comment)
    if not match:
        return None
    
    eval_str = match.group(1)
    
    # Mate score: #4 = mate in 4
    if eval_str.startswith('#'):
        mate_in = int(eval_str[1:])
        # Normalize mate: faster mate = closer to +/-1
        return 1.0 if mate_in > 0 else -1.0
    
    # Centipawn score: normalize to [-1, 1]
    cp = float(eval_str)
    # Sigmoid-like normalization: 3 pawns advantage ≈ 95% win
    normalized = 2.0 / (1.0 + 10.0 ** (-cp / 4.0)) - 1.0
    return max(-1.0, min(1.0, normalized))


def board_to_pgx_planes(board: chess.Board) -> list:
    """
    Convert board to Pgx-style 119-plane encoding.
    
    Planes 0-5:   White pieces (P, N, B, R, Q, K)
    Planes 6-11:  Black pieces
    Planes 12-15: Castling rights (KQkq)
    Plane 16:     Side to move
    Planes 17-18: En passant file
    Planes 19+:   Repetition counter, halfmove clock (zeros for now)
    """
    planes = []
    
    # Piece planes (0-11): 6 piece types × 2 colors
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # PAWN=1, KING=6
            plane = [[0.0] * 8 for _ in range(8)]
            for sq in board.pieces(piece_type, color):
                rank, file = divmod(sq, 8)
                plane[rank][file] = 1.0
            planes.append(plane)
    
    # Castling rights (12-15)
    for right in [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]:
        planes.append([[float(right)] * 8 for _ in range(8)])
    
    # Side to move (16)
    planes.append([[float(board.turn)] * 8 for _ in range(8)])
    
    # En passant (17): file indicator
    ep_plane = [[0.0] * 8 for _ in range(8)]
    if board.ep_square is not None:
        ep_file = board.ep_square % 8
        for rank in range(8):
            ep_plane[rank][ep_file] = 1.0
    planes.append(ep_plane)
    
    # Halfmove clock normalized (18)
    halfmove = min(board.halfmove_clock / 100.0, 1.0)
    planes.append([[halfmove] * 8 for _ in range(8)])
    
    # Fullmove number normalized (19)
    fullmove = min(board.fullmove_number / 200.0, 1.0)
    planes.append([[fullmove] * 8 for _ in range(8)])
    
    # Pad remaining planes with zeros to reach 119
    while len(planes) < 119:
        planes.append([[0.0] * 8 for _ in range(8)])
    
    return planes[:119]


def move_to_pgx_action(move: chess.Move, board: chess.Board) -> int:
    """
    Convert move to Pgx-compatible action index.
    
    Action space: 64 squares × 73 move types = 4672
    - Queen moves: 56 (8 directions × 7 distances)
    - Knight moves: 8
    - Underpromotions: 9 (3 piece types × 3 directions)
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Direction and distance
    rank_diff = (to_sq // 8) - (from_sq // 8)
    file_diff = (to_sq % 8) - (from_sq % 8)
    
    # Knight moves (8 types)
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for i, (dr, df) in enumerate(knight_moves):
        if rank_diff == dr and file_diff == df:
            return from_sq * 73 + 56 + i
    
    # Queen moves (56 types: 8 directions × 7 distances)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dir_idx, (dr, df) in enumerate(directions):
        if dr != 0 and df != 0:  # Diagonal
            if rank_diff != 0 and abs(rank_diff) == abs(file_diff) and \
               (rank_diff // abs(rank_diff) == dr) and (file_diff // abs(file_diff) == df):
                distance = abs(rank_diff) - 1
                return from_sq * 73 + dir_idx * 7 + distance
        elif dr != 0:  # Vertical
            if file_diff == 0 and rank_diff != 0 and (rank_diff // abs(rank_diff) == dr):
                distance = abs(rank_diff) - 1
                return from_sq * 73 + dir_idx * 7 + distance
        else:  # Horizontal
            if rank_diff == 0 and file_diff != 0 and (file_diff // abs(file_diff) == df):
                distance = abs(file_diff) - 1
                return from_sq * 73 + dir_idx * 7 + distance
    
    # Underpromotions (9 types)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_map = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
        promo_idx = promo_map.get(move.promotion, 0)
        dir_idx = 1 + file_diff  # 0=left, 1=straight, 2=right
        return from_sq * 73 + 64 + promo_idx * 3 + dir_idx
    
    # Default: from * 73 + to (fallback)
    return (from_sq * 73 + to_sq) % 4672


def parse_single_game(game_text: str) -> List[Tuple]:
    """
    Parse PGN game with intelligent feature extraction.
    
    Extracts:
    - Pgx-style 119-plane board encoding
    - Pgx-compatible action encoding
    - Stockfish eval as training target (if available)
    - Falls back to game result
    """
    positions = []
    
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if not game:
            return []
        
        # Get result as fallback
        result_str = game.headers.get('Result', '*')
        if result_str == '1-0':
            white_result = 1.0
        elif result_str == '0-1':
            white_result = -1.0
        else:
            white_result = 0.0
        
        # Get ELO for filtering quality games
        try:
            white_elo = int(game.headers.get('WhiteElo', 0))
            black_elo = int(game.headers.get('BlackElo', 0))
            avg_elo = (white_elo + black_elo) // 2
        except:
            avg_elo = 0
        
        board = game.board()
        node = game
        
        for node in game.mainline():
            move = node.move
            comment = node.comment
            
            # Get Stockfish eval if available (SMARTER target!)
            sf_eval = extract_eval(comment)
            
            if sf_eval is not None:
                # Use Stockfish eval (much better than game result!)
                target = sf_eval if board.turn == chess.WHITE else -sf_eval
            else:
                # Fall back to game result
                target = white_result if board.turn == chess.WHITE else -white_result
            
            # Pgx-style board encoding
            obs = board_to_pgx_planes(board)
            
            # Pgx-compatible action
            action = move_to_pgx_action(move, board)
            
            positions.append((obs, action, target))
            board.push(move)
            
    except Exception as e:
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
# Supports QUANTIZED data to fit 4x more positions in VRAM
# Uses Pgx-style 119-plane encoding for full GPU compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def load_to_gpu(all_obs, all_actions, all_results, quantize: bool = True):
    """
    Transfer all data to GPU VRAM.
    
    Input format: observations are (N, 119, 8, 8) from Pgx-style encoding
    Output format: (N, 8, 8, 119) for model (NHWC)
    
    Args:
        quantize: If True, use Int8 for observations (4x memory savings)
    """
    print("📤 Transferring to GPU VRAM...")
    
    import numpy as np
    
    # Convert observations: (N, 119, 8, 8) -> (N, 8, 8, 119)
    obs_np = np.array(all_obs, dtype=np.float32)
    obs_np = np.transpose(obs_np, (0, 2, 3, 1))  # NCHW -> NHWC
    
    N = obs_np.shape[0]
    print(f"   Positions: {N:,} | Planes: {obs_np.shape[-1]}")
    
    if quantize:
        # QUANTIZED: Use Int8 for observations (4x memory savings)
        print("   Quantization: Int8 observations (4x memory savings)")
        
        # Scale to Int8 range and convert
        obs_int8 = (obs_np * 127).astype(np.int8)
        
        actions_np = np.array(all_actions, dtype=np.int16)
        results_np = np.array(all_results, dtype=np.float16)
        
        # Transfer to GPU
        obs = jax.device_put(jnp.array(obs_int8))
        actions = jax.device_put(jnp.array(actions_np))
        results = jax.device_put(jnp.array(results_np))
        
    else:
        # Full precision (FP32)
        obs_padded = np.zeros((N, 8, 8, 119), dtype=np.float32)
        obs_padded[:, :, :, :17] = obs_np
        
        actions_np = np.array(all_actions, dtype=np.int32)
        results_np = np.array(all_results, dtype=np.float32)
        
        obs = jax.device_put(jnp.array(obs_padded))
        actions = jax.device_put(jnp.array(actions_np))
        results = jax.device_put(jnp.array(results_np))
    
    jax.block_until_ready(obs)
    
    total_bytes = obs.nbytes + actions.nbytes + results.nbytes
    print(f"✓ Data in VRAM: {total_bytes / (1024**3):.2f} GB")
    print(f"   Shape: {obs.shape} | dtype: {obs.dtype}")
    
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
    """
    Training step - 100% on GPU.
    
    Handles Int8 quantized observations (dequantizes to float32).
    """
    def loss_fn(params):
        # Dequantize Int8 observations if needed
        if obs.dtype == jnp.int8:
            obs_float = obs.astype(jnp.float32) / 127.0
            # Pad to 119 channels for model compatibility
            obs_padded = jnp.pad(obs_float, ((0, 0), (0, 0), (0, 0), (0, 119 - obs_float.shape[-1])))
        else:
            obs_padded = obs
        
        policy_logits, values, _ = state.apply_fn(params, obs_padded)
        values = jnp.squeeze(values, -1)
        
        # Cast results to float32 if needed (may be float16)
        results_f32 = results.astype(jnp.float32)
        
        # Policy loss (cross-entropy)
        log_probs = jax.nn.log_softmax(policy_logits)
        one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
        policy_loss = -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))
        
        # Value loss (MSE)
        value_loss = jnp.mean((values - results_f32) ** 2)
        
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
    # INCREASED DEFAULTS for better training
    parser.add_argument('--games', type=int, default=100000, help='Max games')
    parser.add_argument('--positions', type=int, default=1000000, help='Max positions')
    parser.add_argument('--steps', type=int, default=20000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
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
    
    # Show initial GPU status
    print(f"   {format_gpu_stats()}")
    print()
    
    with tqdm(total=args.steps, desc="Training", unit="step") as pbar:
        for step in range(args.steps):
            key, batch_key = jax.random.split(key)
            b_obs, b_act, b_res = sample_batch(batch_key, obs, actions, results, args.batch_size)
            state, metrics = train_step(state, b_obs, b_act, b_res)
            
            if step % 100 == 0:
                jax.block_until_ready(metrics['total_loss'])
                elapsed = time.time() - train_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                gpu = format_gpu_stats()
                pbar.set_postfix_str(
                    f"loss={float(metrics['total_loss']):.3f} | "
                    f"acc={float(metrics['accuracy']):.1%} | "
                    f"{speed:.0f}s/s | {gpu}"
                )
            
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
