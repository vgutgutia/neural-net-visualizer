# Neural Network Visualizer

Watch a neural network learn in real-time. Configure architecture, dataset, and hyperparameters — then watch the decision boundary form as training progresses.

**Great for:** understanding how depth, learning rate, and activation functions affect learning.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:5002
```

## Features

- Toy datasets: Circles, Moons, Spiral, XOR
- Configurable hidden layers and activation functions
- Live loss/accuracy curves (Plotly)
- Decision boundary visualization

## Running on necron (RTX 5080)

Training uses CPU by default (fast enough for toy datasets). For GPU:

```bash
# In app.py, change device to:
device = torch.device("cuda")
```

Then SSH in via Tailscale and run as usual.

## Tech Stack

Python · Flask · PyTorch · Plotly · scikit-learn
