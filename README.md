# Neural Net Visualizer

Design a neural network in the browser and watch it learn, live.

**Live demo:** [neural.spbdatascience.org](https://neural.spbdatascience.org)

## Features

- Interactive architecture builder: add and remove hidden layers, resize each layer
- Five synthetic datasets (moons, circles, spiral, XOR, linear) with a noise slider and instant preview
- Live decision boundary, loss, and accuracy streamed during training over server-sent events
- Choice of optimizer (Adam, SGD with momentum, RMSprop) and activation (ReLU, tanh, sigmoid, leaky ReLU)
- SVG diagram of the current architecture, updated as you edit

## How it works

Training runs server-side in PyTorch. The `/api/train_stream` endpoint is a generator that yields SSE events from inside the training loop: scalar metrics every few epochs and a softmax decision-boundary grid several times per run. The response sets `X-Accel-Buffering: no` so nginx does not buffer the stream, and the client reads it with `fetch` + `ReadableStream` (POST bodies rule out `EventSource`).

## Stack

Python, PyTorch, Flask, server-sent events, Plotly

## Local development

```bash
pip install flask torch scikit-learn numpy
python app.py
```
