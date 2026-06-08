import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)


def make_model(layers: list[int], activation: str) -> nn.Sequential:
    act_map = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "sigmoid":    nn.Sigmoid,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }
    Act = act_map.get(activation, nn.ReLU)
    modules: list[nn.Module] = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(Act() if callable(Act) and Act is not nn.ReLU else Act())
    return nn.Sequential(*modules)


def make_optimizer(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def get_dataset(name: str):
    np.random.seed(42)
    n = 300
    if name == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.4)
    elif name == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n, noise=0.15)
    elif name == "xor":
        X = np.random.randn(n, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
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
    layers     = [2] + cfg.get("hidden", [8, 8]) + [2]
    lr         = float(cfg.get("lr", 0.01))
    epochs     = int(cfg.get("epochs", 300))
    dataset    = cfg.get("dataset", "moons")
    activation = cfg.get("activation", "relu")
    optimizer  = cfg.get("optimizer", "adam")

    X, y = get_dataset(dataset)
    model   = make_model(layers, activation)
    opt     = make_optimizer(optimizer, model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    log_every = max(1, epochs // 60)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(1)
                acc   = (preds == y).float().mean().item()
            history.append({
                "epoch": epoch,
                "loss":  round(loss.item(), 4),
                "acc":   round(acc, 4),
            })

    # Decision boundary grid
    res = 70
    lin = np.linspace(-3, 3, res)
    xx, yy = np.meshgrid(lin, lin)
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        zz = torch.softmax(model(grid), dim=1)[:, 1].numpy().reshape(res, res)

    c0_mask = (y == 0).numpy()
    c1_mask = (y == 1).numpy()
    Xnp = X.numpy()

    boundary_fig = go.Figure(data=[
        go.Contour(
            x=lin, y=lin, z=zz,
            colorscale=[[0, "#3d001f"], [0.5, "#1a1a2e"], [1, "#003366"]],
            showscale=False, opacity=0.6,
            contours=dict(coloring="fill"),
        ),
        go.Scatter(
            x=Xnp[c0_mask, 0].tolist(), y=Xnp[c0_mask, 1].tolist(),
            mode="markers", name="Class 0",
            marker=dict(color="#8b0044", size=5, line=dict(color="#0d0d0d", width=.5)),
        ),
        go.Scatter(
            x=Xnp[c1_mask, 0].tolist(), y=Xnp[c1_mask, 1].tolist(),
            mode="markers", name="Class 1",
            marker=dict(color="#4a90d9", size=5, line=dict(color="#0d0d0d", width=.5)),
        ),
    ])
    boundary_fig.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#141414",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.12, font=dict(color="#888")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    encoder = plotly.utils.PlotlyJSONEncoder()
    boundary_json = json.loads(encoder.encode(boundary_fig))

    return jsonify({
        "history":    history,
        "boundary":   boundary_json,
        "final_loss": round(history[-1]["loss"], 4),
        "final_acc":  round(history[-1]["acc"], 4),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
