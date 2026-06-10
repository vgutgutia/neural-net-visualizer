import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

GRID_RES = 55  # resolution for decision boundary grid


def make_model(layers: list, activation: str) -> nn.Sequential:
    act_map = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "sigmoid":    nn.Sigmoid,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }
    Act = act_map.get(activation, nn.ReLU)
    modules = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(Act())
    return nn.Sequential(*modules)


def make_optimizer(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def get_dataset(name: str, noise: float = 0.12, n: int = 300):
    np.random.seed(42)
    if name == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n, noise=noise, factor=0.4)
    elif name == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n, noise=noise)
    elif name == "xor":
        X = np.random.randn(n, 2)
        X += np.random.randn(n, 2) * noise
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif name == "linear":
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n) * noise > 0).astype(int)
    else:  # spiral
        theta = np.linspace(0, 4 * np.pi, n // 2)
        r     = np.linspace(0.1, 1, n // 2)
        X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
        X  = np.vstack([X1, X2]) + np.random.randn(n, 2) * noise
        y  = np.array([0] * (n // 2) + [1] * (n // 2))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.FloatTensor(X), torch.LongTensor(y)


def scatter_payload(X: torch.Tensor, y: torch.Tensor) -> dict:
    """Return lightweight scatter data for the client."""
    Xnp = X.numpy()
    ynp = y.numpy()
    c0  = ynp == 0
    return {
        "x0": Xnp[c0,  0].round(4).tolist(),
        "y0": Xnp[c0,  1].round(4).tolist(),
        "x1": Xnp[~c0, 0].round(4).tolist(),
        "y1": Xnp[~c0, 1].round(4).tolist(),
    }


def compute_boundary(model: nn.Module, lin: np.ndarray) -> list:
    """Return 2-D softmax grid (class-1 probability) as nested list."""
    xx, yy = np.meshgrid(lin, lin)
    grid   = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    with torch.no_grad():
        zz = torch.softmax(model(grid), dim=1)[:, 1].numpy()
    return zz.reshape(len(lin), len(lin)).round(4).tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    cfg   = request.get_json(force=True)
    X, y  = get_dataset(cfg.get("dataset", "moons"), float(cfg.get("noise", 0.12)))
    return jsonify(scatter_payload(X, y))


@app.route("/api/train_stream", methods=["POST"])
def train_stream():
    cfg        = request.get_json(force=True)
    hidden     = [max(1, min(128, int(n))) for n in cfg.get("hidden", [8, 8])]
    layers     = [2] + hidden + [2]
    lr         = float(cfg.get("lr", 0.01))
    epochs     = max(50, min(1000, int(cfg.get("epochs", 300))))
    dataset    = cfg.get("dataset", "moons")
    noise      = float(cfg.get("noise", 0.12))
    activation = cfg.get("activation", "relu")
    optimizer  = cfg.get("optimizer", "adam")

    X, y    = get_dataset(dataset, noise)
    model   = make_model(layers, activation)
    opt     = make_optimizer(optimizer, model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    lin     = np.linspace(-3, 3, GRID_RES)

    log_every      = max(1, epochs // 80)    # ~80 loss/accuracy points
    boundary_every = max(1, epochs // 6)     # ~6 decision boundary snapshots

    def generate():
        # First event: dataset scatter so the client can show it immediately
        scatter = scatter_payload(X, y)
        yield f"data: {json.dumps({'type': 'preview', **scatter})}\n\n"

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out  = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            is_last      = epoch == epochs - 1
            send_log     = (epoch % log_every == 0) or is_last
            send_boundary = (epoch % boundary_every == 0) or is_last

            if send_log:
                model.eval()
                with torch.no_grad():
                    preds = model(X).argmax(1)
                    acc   = float((preds == y).float().mean().item())

                event = {
                    "type":  "update",
                    "epoch": epoch,
                    "loss":  round(float(loss.item()), 4),
                    "acc":   round(acc, 4),
                }
                if send_boundary:
                    event["boundary"] = compute_boundary(model, lin)
                    event["lin"]      = lin.round(4).tolist()

                yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
