#!/usr/bin/env python3
"""
Lichess Database Training for RecurseZero.

Downloads official Lichess database, processes millions of games,
and trains entirely on GPU using JAX for maximum speed.

Usage:
    python train_lichess.py                          # Default: 100k games
    python train_lichess.py --games 1000000          # 1M games
    python train_lichess.py --month 2019-09          # Specific month
"""

import os
import sys
import time
import pickle
import argparse
import subprocess
from typing import List, Tuple, Generator
from functools import partial

print("=" * 60)
print("🎯 RecurseZero - Lichess Human Data Training")
print("=" * 60)
print()

# Set XLA flags before JAX import
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import optax

print(f"✓ JAX loaded (backend: {jax.default_backend()})")

# Auto-install dependencies
def install_dependencies():
    """Install required packages."""
    packages = ['chess', 'zstandard', 'requests']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install_dependencies()

import chess
import chess.pgn
import zstandard
import requests
import io


# ═══════════════════════════════════════════════════════════════════════════════
# LICHESS DATABASE DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
DATA_DIR = "lichess_data"


def download_lichess_database(month: str = "2019-09", max_size_gb: float = 5.0) -> str:
    """
    Download Lichess database for a specific month.
    
    Args:
        month: Month in YYYY-MM format (e.g., "2019-09")
        max_size_gb: Maximum download size in GB
        
    Returns:
        Path to downloaded file
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    url = LICHESS_DB_URL.format(month=month)
    filepath = os.path.join(DATA_DIR, f"lichess_{month}.pgn.zst")
    
    # Check if already downloaded
    if os.path.exists(filepath):
        size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"✓ Found cached database: {filepath} ({size_gb:.2f} GB)")
        return filepath
    
    print(f"📥 Downloading Lichess database: {month}")
    print(f"   URL: {url}")
    print(f"   This may take a while for large files...")
    
    try:
        # Stream download with progress
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        total_gb = total_size / (1024**3)
        
        if total_gb > max_size_gb:
            print(f"   ⚠ File is {total_gb:.1f} GB, limiting download to {max_size_gb} GB")
        
        downloaded = 0
        max_bytes = int(max_size_gb * 1024**3)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress
                    if downloaded % (100 * 1024 * 1024) == 0:  # Every 100MB
                        print(f"   Downloaded: {downloaded / (1024**3):.2f} GB", end='\r')
                    
                    if downloaded >= max_bytes:
                        print(f"\n   Reached {max_size_gb} GB limit, stopping download")
                        break
        
        print(f"\n✓ Downloaded: {filepath} ({downloaded / (1024**3):.2f} GB)")
        return filepath
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print("   Using sample games instead...")
        return None


def stream_pgn_games(filepath: str, max_games: int = 100000) -> Generator[chess.pgn.Game, None, None]:
    """
    Stream PGN games from zstd-compressed file.
    
    Yields:
        chess.pgn.Game objects
    """
    if filepath is None or not os.path.exists(filepath):
        # Return sample games
        for game in get_sample_games(max_games):
            yield game
        return
    
    print(f"📖 Reading games from {filepath}...")
    
    dctx = zstandard.ZstdDecompressor()
    games_read = 0
    
    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            while games_read < max_games:
                try:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    # Skip very short games
                    if len(list(game.mainline_moves())) < 10:
                        continue
                    
                    games_read += 1
                    
                    if games_read % 10000 == 0:
                        print(f"   Processed: {games_read:,} games", end='\r')
                    
                    yield game
                    
                except Exception:
                    continue
    
    print(f"\n✓ Read {games_read:,} games")


def get_sample_games(n: int = 1000) -> List[chess.pgn.Game]:
    """Generate sample games for testing."""
    games = []
    sample_pgns = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 c6 1-0",
        "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 1/2-1/2",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 0-1",
    ]
    
    for i in range(n):
        pgn = sample_pgns[i % len(sample_pgns)]
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game:
            games.append(game)
    
    return games


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-NATIVE DATA PROCESSING (Pgx-style encoding)
# ═══════════════════════════════════════════════════════════════════════════════

def board_to_planes(board: chess.Board) -> jnp.ndarray:
    """
    Convert chess board to neural network input (Pgx-style encoding).
    
    Format: 8x8x119 planes (AlphaZero format)
    - Planes 0-5: White pieces (P, N, B, R, Q, K)
    - Planes 6-11: Black pieces
    - Planes 12-15: Castling rights
    - Plane 16: Side to move
    - Planes 17-118: History/repetition (zeros for now)
    """
    planes = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    
    # Piece planes
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            rank, file = divmod(sq, 8)
            plane_idx = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6
            planes = planes.at[rank, file, plane_idx].set(1.0)
    
    # Castling rights (planes 12-15)
    planes = planes.at[:, :, 12].set(float(board.has_kingside_castling_rights(chess.WHITE)))
    planes = planes.at[:, :, 13].set(float(board.has_queenside_castling_rights(chess.WHITE)))
    planes = planes.at[:, :, 14].set(float(board.has_kingside_castling_rights(chess.BLACK)))
    planes = planes.at[:, :, 15].set(float(board.has_queenside_castling_rights(chess.BLACK)))
    
    # Side to move (plane 16)
    planes = planes.at[:, :, 16].set(float(board.turn))
    
    return planes


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """
    Convert chess move to action index (Pgx-compatible).
    
    Action space: 64 * 73 = 4672 (from_square * move_type)
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Simple encoding: from_square * 64 + direction_offset
    # This is simplified; full Pgx uses 73 move types per square
    
    rank_diff = (to_sq // 8) - (from_sq // 8)
    file_diff = (to_sq % 8) - (from_sq % 8)
    
    # Encode direction (simplified)
    if move.promotion:
        # Promotion moves
        promo_offset = {chess.QUEEN: 64, chess.ROOK: 65, chess.BISHOP: 66, chess.KNIGHT: 67}
        action = from_sq * 73 + promo_offset.get(move.promotion, 64)
    else:
        # Regular moves - encode as from * 73 + to
        action = from_sq * 73 + (to_sq % 73)
    
    return action % 4672  # Clamp to action space


def process_game_to_positions(game: chess.pgn.Game) -> List[Tuple[jnp.ndarray, int, float]]:
    """
    Convert a PGN game to list of (observation, action, result) tuples.
    
    All tensors are ready for GPU transfer.
    """
    positions = []
    
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
        # Get observation
        obs = board_to_planes(board)
        
        # Get action
        action = move_to_action(move, board)
        
        # Result from current player's perspective
        result = white_result if board.turn == chess.WHITE else -white_result
        
        positions.append((obs, action, result))
        board.push(move)
    
    return positions


def create_gpu_dataset(
    games: Generator,
    max_positions: int = 1000000,
    batch_size: int = 4096
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Process games and load entire dataset into GPU VRAM.
    
    Returns:
        obs: (N, 8, 8, 119) observations on GPU
        actions: (N,) action indices on GPU
        results: (N,) game results on GPU
    """
    print(f"🔄 Processing games into GPU-native format...")
    
    all_obs = []
    all_actions = []
    all_results = []
    
    total_positions = 0
    games_processed = 0
    
    for game in games:
        positions = process_game_to_positions(game)
        
        for obs, action, result in positions:
            all_obs.append(obs)
            all_actions.append(action)
            all_results.append(result)
            total_positions += 1
            
            if total_positions >= max_positions:
                break
        
        games_processed += 1
        
        if total_positions >= max_positions:
            break
        
        if games_processed % 1000 == 0:
            print(f"   Games: {games_processed:,} | Positions: {total_positions:,}", end='\r')
    
    print(f"\n✓ Processed {games_processed:,} games, {total_positions:,} positions")
    
    # Stack and transfer to GPU
    print("📤 Transferring to GPU VRAM...")
    
    obs = jnp.stack(all_obs)
    actions = jnp.array(all_actions, dtype=jnp.int32)
    results = jnp.array(all_results, dtype=jnp.float32)
    
    # Force transfer to GPU
    obs = jax.device_put(obs)
    actions = jax.device_put(actions)
    results = jax.device_put(results)
    
    # Wait for transfer
    jax.block_until_ready(obs)
    
    # Calculate memory usage
    total_bytes = obs.nbytes + actions.nbytes + results.nbytes
    print(f"✓ Dataset in VRAM: {total_bytes / (1024**3):.2f} GB")
    print(f"   Observations: {obs.shape}")
    print(f"   Actions: {actions.shape}")
    print(f"   Results: {results.shape}")
    
    return obs, actions, results


# ═══════════════════════════════════════════════════════════════════════════════
# GPU TRAINING (100% on GPU)
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=['batch_size'])
def get_batch(key, obs, actions, results, batch_size):
    """Sample random batch from GPU dataset (no CPU transfer)."""
    n = obs.shape[0]
    indices = jax.random.choice(key, n, shape=(batch_size,), replace=False)
    return obs[indices], actions[indices], results[indices]


@jax.jit
def train_step(state, obs, actions, results):
    """
    Supervised training step - 100% on GPU.
    
    Losses:
    - Policy: Cross-entropy with human moves
    - Value: MSE with game result
    """
    def loss_fn(params):
        policy_logits, values, _ = state.apply_fn(params, obs)
        values = jnp.squeeze(values, -1)
        
        # Policy loss
        log_probs = jax.nn.log_softmax(policy_logits)
        one_hot = jax.nn.one_hot(actions, policy_logits.shape[-1])
        policy_loss = -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))
        
        # Value loss
        value_loss = jnp.mean((values - results) ** 2)
        
        # Accuracy
        predicted_moves = jnp.argmax(policy_logits, axis=-1)
        accuracy = jnp.mean(predicted_moves == actions)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        return total_loss, {
            'policy_loss': policy_loss, 
            'value_loss': value_loss,
            'accuracy': accuracy
        }
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, {'total_loss': loss, **aux}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train on Lichess database')
    parser.add_argument('--month', type=str, default='2019-09', help='Lichess database month (YYYY-MM)')
    parser.add_argument('--games', type=int, default=50000, help='Max games to process')
    parser.add_argument('--positions', type=int, default=500000, help='Max positions to load')
    parser.add_argument('--steps', type=int, default=10000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--max_download_gb', type=float, default=2.0, help='Max download size in GB')
    parser.add_argument('--output', type=str, default='lichess_model.pkl', help='Output model path')
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Month: {args.month}")
    print(f"  Max games: {args.games:,}")
    print(f"  Max positions: {args.positions:,}")
    print(f"  Training steps: {args.steps:,}")
    print(f"  Batch size: {args.batch_size:,}")
    
    # Download database
    print()
    print("=" * 60)
    print("DATA DOWNLOAD")
    print("=" * 60)
    
    db_path = download_lichess_database(args.month, args.max_download_gb)
    
    # Process into GPU dataset
    print()
    print("=" * 60)
    print("GPU DATA LOADING")
    print("=" * 60)
    
    games = stream_pgn_games(db_path, args.games)
    obs, actions, results = create_gpu_dataset(games, args.positions)
    
    # Initialize model
    print()
    print("=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    
    from model.agent import RecurseZeroAgentSimple
    from flax.training import train_state
    
    agent = RecurseZeroAgentSimple(num_actions=4672)
    
    key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    
    key, init_key = jax.random.split(key)
    params = agent.init(init_key, dummy_obs)
    
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model: {param_count:,} parameters")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    )
    
    class TrainState(train_state.TrainState):
        pass
    
    state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=optimizer
    )
    
    # JIT warmup
    print("Compiling training step...")
    key, batch_key = jax.random.split(key)
    batch_obs, batch_actions, batch_results = get_batch(batch_key, obs, actions, results, args.batch_size)
    state, _ = train_step(state, batch_obs, batch_actions, batch_results)
    jax.block_until_ready(state.step)
    print("✓ JIT compiled")
    
    # Training
    print()
    print("=" * 60)
    print(f"TRAINING ({args.steps:,} steps)")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    for step in range(args.steps):
        key, batch_key = jax.random.split(key)
        batch_obs, batch_actions, batch_results = get_batch(batch_key, obs, actions, results, args.batch_size)
        state, metrics = train_step(state, batch_obs, batch_actions, batch_results)
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step:5d} │ Loss: {float(metrics['total_loss']):.4f} │ "
                  f"Policy: {float(metrics['policy_loss']):.4f} │ "
                  f"Accuracy: {float(metrics['accuracy']):.1%} │ "
                  f"{steps_per_sec:.0f} s/s")
    
    # Save model
    print()
    print("=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60)
    
    from training.checkpoint import export_for_inference
    export_for_inference(state.params, args.output)
    
    total_time = time.time() - start_time
    print()
    print(f"✅ Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final accuracy: {float(metrics['accuracy']):.1%}")
    print(f"  Model saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
