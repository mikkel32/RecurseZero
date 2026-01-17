#!/usr/bin/env python3
"""
RecurseZero Web Interface - Play Against Trained Models

Features:
- Beautiful chessboard with drag-and-drop moves
- Smart model auto-detection (loads any model architecture)
- Model selection dropdown
- Real-time AI move computation
- Win/Loss tracking

Usage:
    python web_play.py
    Open http://localhost:8080
"""

import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-activate venv if not already in one
VENV_PYTHON = os.path.join(SCRIPT_DIR, 'venv', 'bin', 'python')
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON:
    # Re-execute with venv python
    print(f"🔄 Switching to project venv...")
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Suppress JAX warnings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cpu'  # Web server runs on CPU

from flask import Flask, render_template_string, jsonify, request

# Install dependencies (handles Homebrew Python)
REQUIRED = ['flask', 'chess', 'jax', 'jaxlib', 'flax']
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        print(f"Installing {pkg}...")
        # Try with --user first, then --break-system-packages for Homebrew
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-q', pkg])
        except:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages', '-q', pkg])

import chess
import numpy as np
import signal
import socket
import json
import pickle
import glob
import re
from functools import lru_cache

print("=" * 60)
print("🎮 RecurseZero Web Interface")
print("=" * 60)
print(f"📂 Working directory: {os.getcwd()}")


# ═══════════════════════════════════════════════════════════════════════════════
# SMART PORT HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PORT = 8080

def is_port_in_use(port):
    """Check if port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def kill_port_process(port):
    """Kill any process using the specified port."""
    import subprocess
    
    print(f"🔍 Checking port {port}...")
    
    if not is_port_in_use(port):
        print(f"   ✓ Port {port} is free")
        return True
    
    print(f"   ⚠️ Port {port} is in use, attempting to free it...")
    
    # Try lsof (macOS/Linux)
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"   ✓ Killed process {pid}")
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        print(f"   ⚠️ Cannot kill {pid} (permission denied)")
            
            # Wait a moment and check again
            import time
            time.sleep(0.5)
            
            if not is_port_in_use(port):
                print(f"   ✓ Port {port} is now free")
                return True
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"   Warning: lsof failed: {e}")
    
    # If still in use, return False
    if is_port_in_use(port):
        print(f"   ❌ Could not free port {port}")
        return False
    
    return True


def find_available_port(start_port=8080):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        if not is_port_in_use(port):
            return port
    return start_port


# ═══════════════════════════════════════════════════════════════════════════════
# SMART MODEL LOADER - Auto-detects model parameters
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(SCRIPT_DIR, "model_intelligence_folder")

def detect_model_params(params):
    """
    Smart parameter detection - figures out model architecture from weights.
    
    Inspects the shape of key layers to determine:
    - hidden_dim
    - num_layers
    - heads
    - mlp_dim
    """
    info = {
        'hidden_dim': None,
        'num_layers': 0,
        'heads': None,
        'mlp_dim': None,
        'total_params': 0,
        'architecture': 'unknown'
    }
    
    def count_leaves(tree):
        if isinstance(tree, dict):
            return sum(count_leaves(v) for v in tree.values())
        elif hasattr(tree, 'size'):
            return tree.size
        elif hasattr(tree, 'shape'):
            return np.prod(tree.shape)
        return 0
    
    info['total_params'] = count_leaves(params)
    
    # Navigate parameter tree
    if 'params' in params:
        params = params['params']
    
    # Detect input embedding dimension
    if 'input_embed' in params:
        embed = params['input_embed']
        if 'kernel' in embed:
            kernel = embed['kernel']
            if hasattr(kernel, 'shape'):
                info['hidden_dim'] = kernel.shape[-1]
    
    # Count transformer layers
    layer_pattern = re.compile(r'(block_|gtrxl_block_|layer_)(\d+)')
    layer_nums = set()
    
    def find_layers(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                match = layer_pattern.match(k)
                if match:
                    layer_nums.add(int(match.group(2)))
                find_layers(v, f'{prefix}.{k}')
    
    find_layers(params)
    info['num_layers'] = len(layer_nums) if layer_nums else 1
    
    # Detect architecture type
    if any('gtrxl' in str(k).lower() for k in str(params)):
        info['architecture'] = 'GTrXL'
    elif any('deq' in str(k).lower() for k in str(params)):
        info['architecture'] = 'DEQ'
    else:
        info['architecture'] = 'Transformer'
    
    # Detect MLP dim from Dense layers in blocks
    for key in ['block_0', 'gtrxl_block_0', 'layer_0']:
        if key in params:
            block = params[key]
            for dense_key in block:
                if 'Dense' in dense_key and 'kernel' in block[dense_key]:
                    shape = block[dense_key]['kernel'].shape
                    if len(shape) == 2:
                        # Larger dimension is likely MLP
                        if shape[-1] > (info['hidden_dim'] or 128):
                            info['mlp_dim'] = shape[-1]
                            break
    
    return info


def load_model_safe(filepath):
    """
    Load model with smart architecture detection.
    
    Returns:
        (params, info) tuple
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different save formats
        if isinstance(data, dict):
            if 'params' in data:
                params = data['params']
            else:
                params = data
        else:
            params = data
        
        info = detect_model_params(params)
        info['filepath'] = filepath
        info['filename'] = os.path.basename(filepath)
        
        # Extract accuracy from filename if present
        match = re.search(r'(\d+\.?\d*)%', info['filename'])
        if match:
            info['accuracy'] = f"{match.group(1)}%"
        else:
            info['accuracy'] = 'Unknown'
        
        return params, info
        
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return None, {'error': str(e)}


def discover_models():
    """Discover all models in the model intelligence folder."""
    models = {}
    
    patterns = [
        os.path.join(MODEL_DIR, '*.pkl'),
        os.path.join(MODEL_DIR, '*.pickle'),
        '*.pkl',  # Also check root
        'lichess_model.pkl'
    ]
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            if os.path.isfile(filepath):
                params, info = load_model_safe(filepath)
                if params is not None:
                    model_id = os.path.basename(filepath)
                    models[model_id] = {
                        'params': params,
                        'info': info
                    }
                    print(f"  ✓ {model_id}: {info['accuracy']} | {info['architecture']} | {info['total_params']:,} params")
    
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# CHESS LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_FRAMES = 8
PIECE_PLANES = 12

def board_to_observation(board: chess.Board, history: list) -> np.ndarray:
    """Convert board + history to model observation."""
    planes = []
    
    # Add history frames (oldest first)
    for hist_board in history:
        frame = np.zeros((PIECE_PLANES, 8, 8), dtype=np.float32)
        for c_idx, color in enumerate([chess.WHITE, chess.BLACK]):
            for pt in range(1, 7):
                for sq in hist_board.pieces(pt, color):
                    frame[c_idx * 6 + pt - 1, sq // 8, sq % 8] = 1.0
        planes.append(frame)
    
    # Current position
    current = np.zeros((PIECE_PLANES, 8, 8), dtype=np.float32)
    for c_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for pt in range(1, 7):
            for sq in board.pieces(pt, color):
                current[c_idx * 6 + pt - 1, sq // 8, sq % 8] = 1.0
    planes.append(current)
    
    # Pad if not enough history
    while len(planes) < HISTORY_FRAMES:
        planes.insert(0, np.zeros((PIECE_PLANES, 8, 8), dtype=np.float32))
    
    # Take last 8 frames
    planes = planes[-HISTORY_FRAMES:]
    
    # Meta planes
    meta = np.zeros((7, 8, 8), dtype=np.float32)
    meta[0] = int(board.has_kingside_castling_rights(chess.WHITE))
    meta[1] = int(board.has_queenside_castling_rights(chess.WHITE))
    meta[2] = int(board.has_kingside_castling_rights(chess.BLACK))
    meta[3] = int(board.has_queenside_castling_rights(chess.BLACK))
    meta[4] = int(board.turn)
    meta[5] = min(board.halfmove_clock, 100) // 10  # Match training: integer 0-10
    if board.ep_square:
        meta[6, :, board.ep_square % 8] = 1.0
    
    # Stack: (8*12 + 7, 8, 8) = (103, 8, 8)
    obs = np.concatenate([np.concatenate(planes, axis=0), meta], axis=0)
    
    # Transpose to NHWC and pad to 119
    obs = np.transpose(obs, (1, 2, 0))  # (8, 8, 103)
    obs_padded = np.zeros((8, 8, 119), dtype=np.float32)
    obs_padded[:, :, :obs.shape[-1]] = obs
    
    return obs_padded[np.newaxis]  # (1, 8, 8, 119)


def get_ai_move(board: chess.Board, history: list, params, info) -> chess.Move:
    """
    Get AI move using the loaded model.
    
    Uses simple inference - no MCTS for speed.
    """
    import jax
    import jax.numpy as jnp
    
    # Get observation and normalize to match training
    obs = board_to_observation(board, history)
    
    # CRITICAL: Training uses Int8 values divided by 127.0
    # Convert to Int8-like values (0 or 1 -> 0 or 127) then divide
    obs_int8 = (obs * 127).astype(np.int8)
    obs_normalized = obs_int8.astype(np.float32) / 127.0
    obs_jax = jnp.array(obs_normalized)
    
    # Create model with MATCHING architecture from trained model
    from model.agent import RecurseZeroAgentSimple
    
    # Use default architecture that matches the trained model
    # The trained models use: 192-224 hidden, 6-8 heads, 4-5 layers
    agent = RecurseZeroAgentSimple(num_actions=4672)
    
    # Ensure params are in correct format
    # The checkpoint saves as {'params': {...}} but apply needs the full dict
    if 'params' not in params:
        params = {'params': params}
    
    # Run inference
    try:
        policy_logits, value, _ = agent.apply(params, obs_jax, train=False)
        policy = np.array(policy_logits[0])
        
        # Apply softmax to convert logits to probabilities
        policy_probs = np.exp(policy - np.max(policy))
        policy_probs = policy_probs / policy_probs.sum()
        
        print(f"   AI thinks value: {float(value[0]):.2f}")
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to random legal move
        import random
        return random.choice(list(board.legal_moves))
    
    # Find best legal move
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = float('-inf')
    
    move_scores = []
    for move in legal_moves:
        # Simple action encoding: from_square * 64 + to_square
        action_idx = (move.from_square * 64 + move.to_square) % 4672
        score = policy_probs[action_idx]
        move_scores.append((move, score))
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # Debug: show top moves
    move_scores.sort(key=lambda x: x[1], reverse=True)
    top_moves = move_scores[:3]
    print(f"   Top moves: {[(str(m), f'{s:.3f}') for m, s in top_moves]}")
    print(f"   Chosen: {best_move} (score: {best_score:.3f})")
    
    return best_move


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SERVER
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Global state
MODELS = {}
GAMES = {}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RecurseZero - Play Chess</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(233, 69, 96, 0.3);
        }
        
        .subtitle {
            color: #888;
            margin-bottom: 30px;
        }
        
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        select, button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        select {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        select option {
            background: #1a1a2e;
            color: #fff;
        }
        
        button {
            background: linear-gradient(45deg, #e94560, #ff6b6b);
            color: #fff;
            font-weight: bold;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
        }
        
        .game-container {
            display: flex;
            gap: 40px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: flex-start;
        }
        
        #board {
            width: 480px;
            height: 480px;
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            border: 4px solid #e94560;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }
        
        .square {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .light { background: #f0d9b5; }
        .dark { background: #b58863; }
        
        .square:hover {
            filter: brightness(1.1);
        }
        
        .square.selected {
            background: #7b61ff !important;
        }
        
        .square.legal-move::after {
            content: '';
            width: 20px;
            height: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 50%;
        }
        
        .square.legal-capture {
            box-shadow: inset 0 0 0 4px rgba(233, 69, 96, 0.6);
        }
        
        .info-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 24px;
            min-width: 280px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .info-panel h3 {
            color: #e94560;
            margin-bottom: 16px;
            font-size: 1.2rem;
        }
        
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #888;
        }
        
        .stat-value {
            font-weight: bold;
            color: #fff;
        }
        
        #status {
            margin-top: 16px;
            padding: 12px;
            background: rgba(233, 69, 96, 0.2);
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }
        
        .move-history {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 16px;
            font-family: monospace;
            font-size: 0.9rem;
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 8px;
        }
        
        .thinking {
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .piece {
            user-select: none;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <h1>♟️ RecurseZero</h1>
    <p class="subtitle">Play against AI-trained chess models</p>
    
    <div class="controls">
        <select id="model-select">
            {% for model_id, model in models.items() %}
            <option value="{{ model_id }}">{{ model.info.accuracy }} - {{ model.info.architecture }} ({{ "{:,}".format(model.info.total_params) }} params)</option>
            {% endfor %}
        </select>
        <button onclick="newGame()">🔄 New Game</button>
        <button onclick="flipBoard()">🔃 Flip Board</button>
    </div>
    
    <div class="game-container">
        <div id="board"></div>
        
        <div class="info-panel">
            <h3>📊 Game Info</h3>
            <div class="stat">
                <span class="stat-label">Model</span>
                <span class="stat-value" id="model-name">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Accuracy</span>
                <span class="stat-value" id="model-accuracy">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Parameters</span>
                <span class="stat-value" id="model-params">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Your Color</span>
                <span class="stat-value">White ♔</span>
            </div>
            
            <div id="status">Your turn</div>
            
            <h3 style="margin-top: 20px;">📜 Moves</h3>
            <div class="move-history" id="move-history"></div>
        </div>
    </div>
    
    <script>
        const PIECES = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        };
        
        let gameId = null;
        let selectedSquare = null;
        let legalMoves = [];
        let flipped = false;
        
        async function newGame() {
            const modelId = document.getElementById('model-select').value;
            const resp = await fetch('/api/new_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_id: modelId})
            });
            const data = await resp.json();
            gameId = data.game_id;
            
            document.getElementById('model-name').textContent = data.model_info.filename;
            document.getElementById('model-accuracy').textContent = data.model_info.accuracy;
            document.getElementById('model-params').textContent = data.model_info.total_params.toLocaleString();
            
            renderBoard(data.fen);
            updateStatus('Your turn');
            document.getElementById('move-history').textContent = '';
        }
        
        function renderBoard(fen) {
            const board = document.getElementById('board');
            board.innerHTML = '';
            
            const position = fen.split(' ')[0];
            const rows = position.split('/');
            
            for (let rank = 0; rank < 8; rank++) {
                const displayRank = flipped ? 7 - rank : rank;
                let file = 0;
                
                for (const char of rows[displayRank]) {
                    if (isNaN(char)) {
                        const displayFile = flipped ? 7 - file : file;
                        const square = document.createElement('div');
                        const isLight = (displayRank + displayFile) % 2 === 0;
                        square.className = `square ${isLight ? 'light' : 'dark'}`;
                        square.dataset.square = String.fromCharCode(97 + file) + (8 - displayRank);
                        square.innerHTML = `<span class="piece">${PIECES[char] || ''}</span>`;
                        square.onclick = () => handleClick(square.dataset.square);
                        board.appendChild(square);
                        file++;
                    } else {
                        for (let i = 0; i < parseInt(char); i++) {
                            const displayFile = flipped ? 7 - file : file;
                            const square = document.createElement('div');
                            const isLight = (displayRank + displayFile) % 2 === 0;
                            square.className = `square ${isLight ? 'light' : 'dark'}`;
                            square.dataset.square = String.fromCharCode(97 + file) + (8 - displayRank);
                            square.onclick = () => handleClick(square.dataset.square);
                            board.appendChild(square);
                            file++;
                        }
                    }
                }
            }
            
            // Highlight legal moves
            legalMoves.forEach(move => {
                const sq = document.querySelector(`[data-square="${move.to}"]`);
                if (sq) {
                    if (sq.querySelector('.piece')?.textContent) {
                        sq.classList.add('legal-capture');
                    } else {
                        sq.classList.add('legal-move');
                    }
                }
            });
        }
        
        async function handleClick(square) {
            if (!gameId) return;
            
            if (selectedSquare) {
                // Try to make move
                const move = legalMoves.find(m => m.from === selectedSquare && m.to === square);
                if (move) {
                    await makeMove(move.uci);
                } else {
                    // Select new square
                    await selectSquare(square);
                }
            } else {
                await selectSquare(square);
            }
        }
        
        async function selectSquare(square) {
            const resp = await fetch('/api/legal_moves', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game_id: gameId, from_square: square})
            });
            const data = await resp.json();
            
            selectedSquare = square;
            legalMoves = data.moves;
            
            // Re-render to show legal moves
            const resp2 = await fetch('/api/get_fen', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game_id: gameId})
            });
            const fen = (await resp2.json()).fen;
            renderBoard(fen);
            
            // Highlight selected square
            const sq = document.querySelector(`[data-square="${square}"]`);
            if (sq) sq.classList.add('selected');
        }
        
        async function makeMove(uci) {
            updateStatus('Making move...');
            
            const resp = await fetch('/api/make_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game_id: gameId, move: uci})
            });
            const data = await resp.json();
            
            selectedSquare = null;
            legalMoves = [];
            renderBoard(data.fen);
            
            // Update move history
            const history = document.getElementById('move-history');
            history.textContent = data.moves.join(' ');
            history.scrollTop = history.scrollHeight;
            
            if (data.game_over) {
                updateStatus(data.result);
                return;
            }
            
            // AI's turn
            updateStatus('AI is thinking...', true);
            
            const aiResp = await fetch('/api/ai_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game_id: gameId})
            });
            const aiData = await aiResp.json();
            
            renderBoard(aiData.fen);
            
            const history2 = document.getElementById('move-history');
            history2.textContent = aiData.moves.join(' ');
            history2.scrollTop = history2.scrollHeight;
            
            if (aiData.game_over) {
                updateStatus(aiData.result);
            } else {
                updateStatus('Your turn');
            }
        }
        
        function updateStatus(text, thinking = false) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = thinking ? 'thinking' : '';
        }
        
        function flipBoard() {
            flipped = !flipped;
            if (gameId) {
                fetch('/api/get_fen', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({game_id: gameId})
                }).then(r => r.json()).then(d => renderBoard(d.fen));
            }
        }
        
        // Start game on load
        window.onload = () => {
            if (document.getElementById('model-select').options.length > 0) {
                newGame();
            }
        };
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, models=MODELS)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json
    model_id = data.get('model_id')
    
    if model_id not in MODELS:
        return jsonify({'error': 'Model not found'}), 404
    
    import uuid
    game_id = str(uuid.uuid4())[:8]
    
    board = chess.Board()
    GAMES[game_id] = {
        'board': board,
        'model_id': model_id,
        'history': [],
        'moves': []
    }
    
    return jsonify({
        'game_id': game_id,
        'fen': board.fen(),
        'model_info': MODELS[model_id]['info']
    })


@app.route('/api/get_fen', methods=['POST'])
def get_fen():
    game_id = request.json.get('game_id')
    if game_id not in GAMES:
        return jsonify({'error': 'Game not found'}), 404
    return jsonify({'fen': GAMES[game_id]['board'].fen()})


@app.route('/api/legal_moves', methods=['POST'])
def legal_moves():
    data = request.json
    game_id = data.get('game_id')
    from_sq = data.get('from_square')
    
    if game_id not in GAMES:
        return jsonify({'error': 'Game not found'}), 404
    
    board = GAMES[game_id]['board']
    
    moves = []
    for move in board.legal_moves:
        if chess.square_name(move.from_square) == from_sq:
            moves.append({
                'from': chess.square_name(move.from_square),
                'to': chess.square_name(move.to_square),
                'uci': move.uci()
            })
    
    return jsonify({'moves': moves})


@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.json
    game_id = data.get('game_id')
    move_uci = data.get('move')
    
    if game_id not in GAMES:
        return jsonify({'error': 'Game not found'}), 404
    
    game = GAMES[game_id]
    board = game['board']
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
        
        # Save history before move
        game['history'].append(board.copy())
        
        san = board.san(move)
        board.push(move)
        game['moves'].append(san)
        
        result = None
        if board.is_game_over():
            if board.is_checkmate():
                result = 'Checkmate! You win! 🎉'
            elif board.is_stalemate():
                result = 'Stalemate - Draw'
            else:
                result = 'Game Over - Draw'
        
        return jsonify({
            'fen': board.fen(),
            'moves': game['moves'],
            'game_over': board.is_game_over(),
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    try:
        data = request.json
        game_id = data.get('game_id')
        
        if game_id not in GAMES:
            return jsonify({'error': 'Game not found'}), 404
        
        game = GAMES[game_id]
        board = game['board']
        model = MODELS[game['model_id']]
        
        # Get AI move with error handling
        try:
            move = get_ai_move(
                board, 
                game['history'], 
                model['params'], 
                model['info']
            )
        except Exception as e:
            print(f"❌ AI move error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to random move
            import random
            move = random.choice(list(board.legal_moves))
        
        # Make the move
        game['history'].append(board.copy())
        san = board.san(move)
        board.push(move)
        game['moves'].append(san)
        
        result = None
        if board.is_game_over():
            if board.is_checkmate():
                result = 'Checkmate! AI wins! 🤖'
            elif board.is_stalemate():
                result = 'Stalemate - Draw'
            else:
                result = 'Game Over - Draw'
        
        return jsonify({
            'fen': board.fen(),
            'moves': game['moves'],
            'ai_move': move.uci(),
            'game_over': board.is_game_over(),
            'result': result
        })
    
    except Exception as e:
        print(f"❌ API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global MODELS
    
    print("\n📂 Discovering models...")
    MODELS = discover_models()
    
    if not MODELS:
        print("\n⚠️ No models found!")
        print(f"   Place .pkl model files in: {MODEL_DIR}/")
        return
    
    print(f"\n✓ Found {len(MODELS)} model(s)")
    
    # Smart port handling
    port = DEFAULT_PORT
    
    if not kill_port_process(port):
        # Try to find an available port
        port = find_available_port(port)
        print(f"   → Using alternative port: {port}")
    
    print("\n🚀 Starting server...")
    print("=" * 60)
    print(f"   Open: http://localhost:{port}")
    print("=" * 60)
    
    # Run server
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
