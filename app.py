#!/usr/bin/env python3
"""
app.py — Flask web UI for the libaix neural network.

Run:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import numpy as np
from flask import Flask, render_template, request, jsonify

from neural_network import NeuralNetwork

app = Flask(__name__)

# ── Datasets ──────────────────────────────────────────────────────────
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
TARGETS = {
    "xor":  np.array([[0], [1], [1], [0]], dtype=np.float64),
    "and":  np.array([[0], [0], [0], [1]], dtype=np.float64),
    "or":   np.array([[0], [1], [1], [1]], dtype=np.float64),
    "nand": np.array([[1], [1], [1], [0]], dtype=np.float64),
}

# ── Global state ──────────────────────────────────────────────────────
models: dict[str, NeuralNetwork] = {}
loss_history: dict[str, list[float]] = {}


def _train(dataset: str, activation: str = "sigmoid", optimizer: str = "sgd",
           lr: float = 1.0, epochs: int = 10_000) -> list[float]:
    nn = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=lr,
                       activation=activation, optimizer=optimizer, seed=42)
    losses = nn.train(INPUTS, TARGETS[dataset], epochs=epochs, log_every=0)
    models[dataset] = nn
    loss_history[dataset] = losses
    return losses


# Train XOR on startup so the UI works immediately
print("Training XOR neural network …")
_train("xor")
print("Training complete!\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    a = max(0, min(1, int(data.get("a", 0))))
    b = max(0, min(1, int(data.get("b", 0))))
    dataset = data.get("dataset", "xor")
    if dataset not in TARGETS:
        return jsonify({"error": "unknown dataset"}), 400
    if dataset not in models:
        _train(dataset)
    nn = models[dataset]
    raw = float(nn.predict(np.array([[a, b]], dtype=np.float64))[0, 0])
    return jsonify({"a": a, "b": b, "raw": round(raw, 6), "result": int(round(raw)),
                    "dataset": dataset, "activation": nn.activation, "optimizer": nn.optimizer})


@app.route("/train", methods=["POST"])
def train_endpoint():
    data = request.get_json(force=True)
    dataset = data.get("dataset", "xor")
    activation = data.get("activation", "sigmoid")
    optimizer = data.get("optimizer", "sgd")
    lr = float(data.get("lr", 1.0))
    epochs = min(int(data.get("epochs", 10_000)), 100_000)
    if dataset not in TARGETS:
        return jsonify({"error": "unknown dataset"}), 400
    losses = _train(dataset, activation, optimizer, lr, epochs)
    # Downsample loss curve for the chart (max 200 points)
    step = max(1, len(losses) // 200)
    sampled = losses[::step]
    preds = models[dataset].predict(INPUTS).tolist()
    expected = TARGETS[dataset].tolist()
    return jsonify({
        "dataset": dataset,
        "activation": activation,
        "optimizer": optimizer,
        "epochs": epochs,
        "final_loss": round(losses[-1], 8),
        "loss_curve": [round(v, 6) for v in sampled],
        "predictions": preds,
        "expected": expected,
    })


@app.route("/datasets", methods=["GET"])
def datasets():
    return jsonify(list(TARGETS.keys()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
