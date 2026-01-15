# RecurseZero

<p align="center">
  <strong>🧠 GPU-Resident Chess RL Agent with Deep Equilibrium Models</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#training">Training</a> •
  <a href="#api">API</a>
</p>

---

## Overview

RecurseZero is a novel chess AI that achieves superhuman performance on consumer hardware by:

- **Eliminating CPU bottleneck**: All computation runs on GPU via JAX-native Pgx
- **O(1) memory depth**: Deep Equilibrium Models provide infinite effective depth
- **Search-free inference**: Muesli algorithm learns instinctive play without MCTS

## Features

| Feature | Description | Spec Reference |
|---------|-------------|----------------|
| 🚀 GPU-Resident | Game logic + neural net on GPU | Section 1.2 |
| ♾️ DEQ Core | Infinite depth, fixed memory | Section 2.2 |
| ⚡ Anderson Acceleration | Fast fixed-point convergence | Section 2.2.1 |
| 🔒 GTrXL Gating | Stable recursive dynamics | Section 2.3 |
| ♟️ Chess Position Bias | 2D relative encodings | Section 2.4 |
| 🎯 Muesli Algorithm | Search-free policy | Section 3.1 |
| 📊 PVE Learning | Value-equivalent representations | Section 3.3 |
| 🔢 Int8 Quantization | 2-4x speedup via AQT | Section 4.1 |
| 📚 Distillation | Teacher-student learning | Section 4.2 |

## Quick Start

### Google Colab (Recommended)

```python
# Install dependencies
!pip install jax[cuda12] flax optax pgx rich aqtp

# Clone repository
!git clone https://github.com/your-repo/RecurseZero.git
%cd RecurseZero

# Run training
!python main.py
```

### Local Setup

```bash
# Clone
git clone https://github.com/your-repo/RecurseZero.git
cd RecurseZero

# Setup environment
./setup.sh

# Run
python main.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RecurseZero Agent                        │
├─────────────────────────────────────────────────────────────┤
│  Input (8×8×119)                                            │
│       ↓                                                     │
│  Dense Embedding → (B, 64, hidden_dim)                      │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Deep Equilibrium Model (DEQ)               │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │     Universal Transformer Block             │    │   │
│  │  │  • ChessformerAttention + Position Bias    │←───┤   │
│  │  │  • GTrXL Gating (stability)                 │    │   │
│  │  │  • QuantizedMLP (Int8)                      │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │         ↑                                           │   │
│  │         └── Anderson Acceleration (fixed-point) ───┘   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  Mean Pooling → (B, hidden_dim)                             │
│       ↓                                                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐                    │
│  │ Policy  │  │  Value  │  │  Reward  │                    │
│  │ (4672)  │  │ [-1,1]  │  │  scalar  │                    │
│  └─────────┘  └─────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Training

### Configuration

| Parameter | Default (Fast) | Full Spec | Description |
|-----------|----------------|-----------|-------------|
| `hidden_dim` | 128 | 256 | Embedding dimension |
| `heads` | 4 | 8 | Attention heads |
| `mlp_dim` | 512 | 1024 | MLP hidden size |
| `deq_iters` | 4 | 8 | DEQ iterations |
| `batch_size` | 2048 | 4096 | Parallel games |

### Metrics

The training loop displays:

- **P(win)**: Win probability from value head (spec 5.2)
- **Conf**: Policy confidence (inverse of entropy)
- **Games**: Total games completed (W/L/D)
- **s/s**: Training steps per second

### Expected Output

```
Step   100 │ Loss: 0.0012 │ P(win): 51.2% │ Conf: 45% │ Games: 1,024 (W:512/L:512/D:0) │ 1.8 s/s
```

## API

### Training

```python
from training.loop import resident_train_step, create_train_state
from model.agent import RecurseZeroAgentFast

agent = RecurseZeroAgentFast(num_actions=4672)
state = create_train_state(agent, params)
state, env_state, metrics = resident_train_step(state, env_state, key, agent.apply, env.step, env._init)
```

### Distillation

```python
from training.distillation import DistillationTrainer, DistillationConfig

config = DistillationConfig(temperature=2.0, alpha=0.7)
trainer = DistillationTrainer(teacher_model, student_model, config)
trainer.load_teacher("teacher.pkl")
```

### Win Probability

```python
from algorithm.pve import value_to_win_probability

value = agent.apply(params, obs)[1]  # Get value head output
win_prob = value_to_win_probability(value)  # Convert to P(win)
```

## File Structure

```
RecurseZero/
├── main.py                 # Entry point with training loop
├── model/
│   ├── agent.py            # RecurseZeroAgent (full & fast variants)
│   ├── deq.py              # Deep Equilibrium Model + Anderson Accel
│   ├── universal_transformer.py  # Transformer block with quantization
│   ├── gtrxl.py            # GTrXL gating mechanism
│   └── embeddings.py       # Chess relative position bias
├── algorithm/
│   ├── muesli.py           # Muesli policy gradient
│   └── pve.py              # Proper Value Equivalence
├── training/
│   ├── loop.py             # GPU-resident training step
│   └── distillation.py     # Teacher-student distillation
├── env/
│   └── pgx_wrapper.py      # GPU-resident Pgx chess
└── optimization/
    ├── quantization.py     # Int8 AQT integration
    └── hardware_compat.py  # MPS/CUDA configuration
```

## References

Based on the "GPU ONLY Chess RL Agent" specification:

1. **Pgx**: Hardware-accelerated game simulators (arXiv:2303.17503)
2. **DEQ**: Deep Equilibrium Models (arXiv:1909.01377)
3. **GTrXL**: Stabilizing Transformers for RL (arXiv:1910.06764)
4. **Muesli**: Search-free policy optimization (arXiv:2104.06159)
5. **AQT**: Accurate Quantized Training (Google Cloud)

## License

MIT License - See LICENSE file for details.
