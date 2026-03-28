#!/usr/bin/env python3
"""
app.py — Flask web UI for the libaix neural network.

Provides:
  • Logic-gate playground (predict, train)
  • AI chat (knowledge Q&A powered by a trained classifier)

Run:
    python app.py
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify

from admin import admin_bp
from knowledge_base import KNOWLEDGE, get_domains
from neural_network import NeuralNetwork
from project_memory import (
    build_startup_context,
    cache_response,
    lookup_cached_response,
    remember,
    update_project_fingerprint,
)
from vectorizer import BagOfWords

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.register_blueprint(admin_bp)

# ── Logic-gate datasets ──────────────────────────────────────────────
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

# Knowledge AI state
knowledge_model: NeuralNetwork | None = None
knowledge_bow: BagOfWords | None = None
knowledge_answer_map: dict[int, str] = {}
knowledge_domains: list[str] = []

MODEL_DIR = Path("models")


def _train(dataset: str, activation: str = "sigmoid", optimizer: str = "sgd",
           lr: float = 1.0, epochs: int = 10_000) -> list[float]:
    nn = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=lr,
                       activation=activation, optimizer=optimizer, seed=42)
    losses = nn.train(INPUTS, TARGETS[dataset], epochs=epochs, log_every=0)
    models[dataset] = nn
    loss_history[dataset] = losses
    return losses


def _load_knowledge_model() -> bool:
    """Load the pre-trained knowledge model, vectorizer, and answer map."""
    global knowledge_model, knowledge_bow, knowledge_answer_map, knowledge_domains
    model_path = MODEL_DIR / "knowledge.npz"
    vec_path = MODEL_DIR / "vectorizer.json"
    ans_path = MODEL_DIR / "answer_map.json"
    if not all(p.exists() for p in (model_path, vec_path, ans_path)):
        return False
    knowledge_model = NeuralNetwork.load(model_path)
    knowledge_bow = BagOfWords.load(vec_path)
    raw = json.loads(ans_path.read_text(encoding="utf-8"))
    knowledge_answer_map = {int(k): v for k, v in raw.items()}
    knowledge_domains = get_domains()
    return True


# Startup
print("Training XOR neural network …")
_train("xor")
print("Training complete!")

if _load_knowledge_model():
    print(f"Knowledge AI loaded — {len(knowledge_answer_map)} answers, "
          f"domains: {', '.join(knowledge_domains)}")
else:
    print("Warning: Knowledge model not found. Run 'python train_knowledge.py' first.")

# Load project memory context
try:
    _ctx = build_startup_context()
    if _ctx.get("project_changed"):
        print("Project files changed since last run — updating fingerprint.")
        update_project_fingerprint()
    else:
        print("Project unchanged — using cached context.")
    remember("project", "structure", {
        "datasets": list(TARGETS.keys()),
        "domains": knowledge_domains,
        "answers": len(knowledge_answer_map),
    })
    if _ctx.get("cache_size", 0) > 0:
        print(f"Response cache: {_ctx['cache_size']} entries loaded.")
    perf = _ctx.get("performance", {})
    if perf.get("entries", 0) > 0:
        print(f"Performance trend: {perf.get('latest_accuracy', 'N/A')} accuracy "
              f"({'↑ improving' if perf.get('improving') else '→ stable'})")
except Exception as _e:
    print(f"Note: Project memory init skipped ({type(_e).__name__}: {_e})")
print()


# ── Routes: pages ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Routes: logic-gate API ────────────────────────────────────────────

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


# ── Routes: Knowledge AI chat ────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """Answer a user question using the knowledge classifier."""
    if knowledge_model is None or knowledge_bow is None:
        return jsonify({"error": "Knowledge model not loaded. Train it first."}), 503

    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Check response cache first
    cached = lookup_cached_response(question)
    if cached and cached.get("confidence", 0) >= 0.15:
        return jsonify({
            "answer": cached["answer"],
            "confidence": cached["confidence"],
            "domain": cached.get("domain", "general"),
            "suggestions": [],
            "cached": True,
        })

    # Vectorize user query
    vec = knowledge_bow.transform([question])

    # Predict
    probs = knowledge_model.predict(vec)[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    answer = knowledge_answer_map.get(top_idx, "I don't have an answer for that.")

    # Find the domain for context
    # Match back to original KNOWLEDGE entry via the answer
    domain = "general"
    for _, a, d in KNOWLEDGE:
        if a == answer:
            domain = d
            break

    # Get top-3 for multi-result
    top3_idx = np.argsort(probs)[::-1][:3]
    suggestions = []
    for idx in top3_idx:
        idx = int(idx)
        a = knowledge_answer_map.get(idx, "")
        if a:
            suggestions.append({
                "answer": a,
                "confidence": round(float(probs[idx]), 4),
            })

    # Low confidence threshold
    if confidence < 0.15:
        answer = ("I'm not sure about that. Try asking about networking, internet, "
                  "intranet, or security topics. Type 'help' for examples.")
        domain = "general"

    # Cache the response for future lookups
    cache_response(question, answer, confidence, domain)

    return jsonify({
        "answer": answer,
        "confidence": round(confidence, 4),
        "domain": domain,
        "suggestions": suggestions,
    })


@app.route("/knowledge/domains", methods=["GET"])
def knowledge_domain_list():
    return jsonify(knowledge_domains)


@app.route("/knowledge/stats", methods=["GET"])
def knowledge_stats():
    return jsonify({
        "loaded": knowledge_model is not None,
        "total_answers": len(knowledge_answer_map),
        "domains": knowledge_domains,
        "vocab_size": knowledge_bow.vocab_size if knowledge_bow else 0,
    })


@app.route("/knowledge/retrain", methods=["POST"])
def retrain_knowledge():
    """Re-train the knowledge model (useful after adding new knowledge)."""
    try:
        from train_knowledge import train as train_knowledge
        train_knowledge(verbose=False)
        ok = _load_knowledge_model()
        if not ok:
            return jsonify({"error": "Training succeeded but model loading failed"}), 500
        return jsonify({
            "success": True,
            "total_answers": len(knowledge_answer_map),
            "domains": knowledge_domains,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/context", methods=["GET"])
def memory_context():
    """Return the current project memory context."""
    try:
        ctx = build_startup_context()
        return jsonify(ctx)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/performance", methods=["GET"])
def memory_performance():
    """Return model performance history."""
    try:
        from project_memory import get_performance_trend
        return jsonify(get_performance_trend(n=20))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
