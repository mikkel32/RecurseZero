# RecurseZero 🧠♟️

**GPU-Resident Chess RL Agent** - Train chess AI entirely on GPU with JAX.

## 🚀 Quick Start

### Self-Play Training (RL)
```bash
python main.py
```

### Lichess Human Data Training
```bash
pip install chess requests
python train_lichess.py --games 10000 --steps 5000
```

## 📊 Performance

| Variant | Speed | Parameters | Use Case |
|---------|-------|------------|----------|
| Simple | **5+ s/s** | 1.2M | Fast training |
| DEQ Fast | 1.8 s/s | 1M | Experimental |
| DEQ Full | 0.5 s/s | 2M | Spec-compliant |

## 📁 Project Structure

```
RecurseZero/
├── main.py                    # Self-play RL training
├── train_lichess.py           # Human data training
├── model/
│   ├── agent.py               # Neural network architectures
│   ├── universal_transformer.py
│   └── deq.py                 # Deep Equilibrium Model
├── training/
│   ├── loop.py                # GPU-resident training step
│   ├── checkpoint.py          # Model saving/loading
│   └── distillation.py        # Knowledge distillation
├── algorithm/
│   ├── muesli.py              # Policy optimization
│   └── pve.py                 # Value estimation
├── env/
│   └── pgx_wrapper.py         # Pgx chess environment
└── optimization/
    └── hardware_compat.py     # JAX/GPU setup
```

## 🎯 Model Saving & Loading

### Save after training
Training automatically saves:
- `checkpoints/recurse_zero_latest.pkl` - Full checkpoint
- `recurse_zero_model.pkl` - Inference-ready export

### Load for inference
```python
from training.checkpoint import load_checkpoint
from model.agent import RecurseZeroAgentSimple

# Load weights
checkpoint = load_checkpoint('checkpoints')
params = checkpoint['params']

# Create model
agent = RecurseZeroAgentSimple(num_actions=4672)
policy, value, _ = agent.apply(params, observation)
```

## 🎮 Training Modes

### Mode 1: Self-Play Reinforcement Learning
```bash
python main.py
```
- Learns from scratch via self-play
- Uses Muesli policy optimization
- No external data needed

### Mode 2: Lichess Human Data
```bash
python train_lichess.py --games 100000 --steps 10000 --output human_model.pkl
```
- Trains on real human games
- Supervised learning (imitation)
- Faster convergence, human-like play

## ⚙️ Configuration

Edit parameters directly in `model/agent.py`:

```python
class RecurseZeroAgentSimple(nn.Module):
    hidden_dim: int = 128    # Model width
    heads: int = 4           # Attention heads  
    mlp_dim: int = 512       # MLP hidden size
    num_layers: int = 3      # Transformer layers
```

## 📈 Training Metrics

| Metric | Description |
|--------|-------------|
| Loss | Combined policy + value loss |
| P(win) | Predicted win probability |
| W/L/D | Games won/lost/drawn |
| s/s | Training steps per second |

## 🔧 Requirements

```
jax[cuda]>=0.4.20
flax>=0.8.0
optax>=0.1.7
pgx>=2.0.0
rich>=13.0.0
chess>=1.10.0  # For Lichess training
```

## 📖 References

- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
- [Muesli: Mastering Atari with Discrete World Models](https://arxiv.org/abs/2104.06159)
- [Pgx: JAX Game Environments](https://github.com/sotetsuk/pgx)
