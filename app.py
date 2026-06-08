"""
Neural Network Visualizer
Watch a neural network learn in real-time.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, Response
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)


def make_model(layers: list[int], activation: str) -> nn.Sequential:
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    Act = act_map.get(activation, nn.ReLU)
    modules = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(Act())
    return nn.Sequential(*modules)


def get_dataset(name: str):
    """Returns X (N,2) and y (N,) tensors for a toy 2D classification dataset."""
    np.random.seed(42)
    n = 200
    if name == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.4)
    elif name == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n, noise=0.15)
    elif name == "xor":
        X = np.random.randn(n, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
    else:  # spiral
        theta = np.linspace(0, 4 * np.pi, n // 2)
        r = np.linspace(0.1, 1, n // 2)
        X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
        X = np.vstack([X1, X2])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.FloatTensor(X), torch.LongTensor(y)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def train():
    cfg = request.get_json(force=True)
    layers    = [2] + cfg.get("hidden", [8, 8]) + [2]
    lr        = float(cfg.get("lr", 0.01))
    epochs    = int(cfg.get("epochs", 100))
    dataset   = cfg.get("dataset", "circles")
    activation = cfg.get("activation", "relu")

    X, y = get_dataset(dataset)
    model = make_model(layers, activation)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        if epoch % max(1, epochs // 50) == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(1)
                acc   = (preds == y).float().mean().item()
            history.append({"epoch": epoch, "loss": round(loss.item(), 4), "acc": round(acc, 4)})

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    grid   = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        zz = torch.softmax(model(grid), dim=1)[:, 1].numpy().reshape(xx.shape)

    boundary_fig = go.Figure(data=[
        go.Contour(x=np.linspace(-3, 3, 60), y=np.linspace(-3, 3, 60), z=zz,
                   colorscale=[[0, "#660033"], [1, "#003366"]],
                   showscale=False, opacity=0.5),
        go.Scatter(x=X[y == 0, 0].numpy(), y=X[y == 0, 1].numpy(),
                   mode="markers", marker=dict(color="#660033", size=5), name="Class 0"),
        go.Scatter(x=X[y == 1, 0].numpy(), y=X[y == 1, 1].numpy(),
                   mode="markers", marker=dict(color="#003366", size=5), name="Class 1"),
    ])
    boundary_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#141414", plot_bgcolor="#141414",
        margin=dict(l=0, r=0, t=0, b=0), height=340,
    )

    return jsonify({
        "history": history,
        "boundary": json.loads(plotly.utils.PlotlyJSONEncoder().encode(boundary_fig)),
        "final_loss": history[-1]["loss"],
        "final_acc":  history[-1]["acc"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5002)
