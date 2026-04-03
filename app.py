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
import re
import secrets
import threading
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify

from admin import admin_bp
from knowledge_base import KNOWLEDGE, get_domains
from neural_network import ACTIVATIONS, OPTIMIZERS, NeuralNetwork
from project_memory import (
    build_startup_context,
    cache_response,
    lookup_cached_response,
    remember,
    update_project_fingerprint,
)
from vectorizer import BagOfWords

# Brain + Watcher (lazy imports — graceful if not yet available)
try:
    from libaix_brain import (
        analyse_impact as _brain_impact,
        build_dependency_graph as _brain_deps,
        build_session_briefing as _brain_briefing,
        detect_stale_data as _brain_stale,
        get_status as _brain_status,
        measure_code_quality as _brain_quality,
        recommend_knowledge_gaps as _brain_knowledge_gaps,
        run_full_scan_cycle as _brain_scan,
        scan_project as _brain_scan_project,
        score_module_complexity as _brain_complexity,
        summarize_module as _brain_module_summary,
        analyse_gaps as _brain_analyse_gaps,
        calculate_health_score as _brain_health_score,
        get_pending_tasks as _brain_pending_tasks,
    )
    from ml_watcher import (
        build_watcher_context as _watcher_context,
        clear_acknowledged_alerts as _watcher_clear_alerts,
        detect_config_drift as _watcher_config_drift,
        get_alert_summary as _watcher_alert_summary,
        measure_disk_usage as _watcher_disk_usage,
        run_health_check as _watcher_health_check,
        run_watcher_cycle as _watcher_cycle,
        track_knowledge_growth as _watcher_growth,
    )
    _BRAIN_AVAILABLE = True
except ImportError:
    _BRAIN_AVAILABLE = False

# Boil engine + Reasoning engine + Anon crawler (lazy imports)
try:
    from boil_engine import (
        get_boil_state,
        get_improvement_log,
        is_boiling,
        load_boil_config,
        run_boil_tick,
        save_boil_config,
        start_boil_background,
        stop_boil_background,
    )
    _BOIL_AVAILABLE = True
except ImportError:
    _BOIL_AVAILABLE = False

try:
    from reasoning_engine import (
        build_reasoning_base as _build_reasoning,
        get_reasoning_engine,
        reason_about,
    )
    _REASONING_AVAILABLE = True
except ImportError:
    _REASONING_AVAILABLE = False

try:
    from anon_crawler import (
        anon_crawl_page,
        anon_crawl_site,
        get_anon_stats,
        load_anon_config,
        save_anon_config,
    )
    _ANON_AVAILABLE = True
except ImportError:
    _ANON_AVAILABLE = False

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

# Brain + Watcher startup
if _BRAIN_AVAILABLE:
    try:
        _brain_scan_project()
        print("LIBAIXBrain: project scan complete.")
    except Exception as _e:
        print(f"Note: Brain startup scan skipped ({type(_e).__name__}: {_e})")

# Boil engine auto-start (continuous background self-improvement)
if _BOIL_AVAILABLE:
    try:
        start_boil_background()
        print("Boil engine: background self-improvement started.")
    except Exception as _e:
        print(f"Note: Boil engine start skipped ({type(_e).__name__}: {_e})")

# Reasoning engine init
if _REASONING_AVAILABLE:
    try:
        _build_reasoning()
        print("Reasoning engine: knowledge base built.")
    except Exception as _e:
        print(f"Note: Reasoning engine init skipped ({type(_e).__name__}: {_e})")
print()


# ── Routes: pages ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Routes: logic-gate API ────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        a = max(0, min(1, int(data.get("a", 0))))
        b = max(0, min(1, int(data.get("b", 0))))
    except (ValueError, TypeError):
        return jsonify({"error": "a and b must be integers (0 or 1)"}), 400
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
    if dataset not in TARGETS:
        return jsonify({"error": "unknown dataset"}), 400
    if activation not in ACTIVATIONS:
        return jsonify({"error": f"activation must be one of {list(ACTIVATIONS)}"}), 400
    if optimizer not in OPTIMIZERS:
        return jsonify({"error": f"optimizer must be one of {list(OPTIMIZERS)}"}), 400
    try:
        lr = float(data.get("lr", 1.0))
        epochs = int(data.get("epochs", 10_000))
    except (ValueError, TypeError):
        return jsonify({"error": "lr must be a number and epochs an integer"}), 400
    lr = max(1e-6, min(lr, 100.0))
    epochs = max(1, min(epochs, 100_000))
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

# Chat command detection patterns
_URL_PATTERN = re.compile(r'https?://[^\s<>"\'`,;)\]]+', re.IGNORECASE)
_RESEARCH_PREFIXES = [
    "research", "learn about", "study", "find information about",
    "teach yourself about", "gather knowledge on", "look up",
    "find out about", "investigate",
]
_CRAWL_ACTION_VERBS = r'^(crawl|scrape|visit|fetch|go to|get|learn from|read)\s+'
MAX_RESEARCH_URLS = 5


def _detect_chat_command(question: str) -> dict | None:
    """Detect research/learn/URL commands in user chat messages.

    Returns a dict with command info, or None for regular questions.
    """
    lower = question.lower().strip()

    # Detect URLs → offer crawling
    urls = _URL_PATTERN.findall(question)
    if urls:
        # Extract topic context from surrounding text
        text = question
        for u in urls:
            text = text.replace(u, "")
        topic = text.strip(" .,;:!?")
        topic = re.sub(
            _CRAWL_ACTION_VERBS,
            '', topic, flags=re.IGNORECASE,
        ).strip() or "general"
        return {"type": "crawl", "urls": urls, "topic": topic}

    # Detect research/learn commands
    for prefix in sorted(_RESEARCH_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            topic = question[len(prefix):].strip(" .,;:!?")
            if len(topic) > 2:
                return {"type": "research", "topic": topic}

    return None


def _execute_research(topic: str, urls: list[str] | None = None) -> dict:
    """Execute multi-source research for a topic. Runs crawlers and returns results.

    This does NOT retrain the model inline — it queues knowledge and triggers
    a background retrain so the response is fast.
    """
    results: dict[str, dict] = {}
    total_entries = 0

    try:
        from crawler import crawl_single_topic
        wiki_result = crawl_single_topic(topic, [], max_articles=5)
        results["wikipedia"] = {
            "entries": wiki_result.get("entries", 0),
            "status": wiki_result.get("status", "error"),
        }
        if wiki_result.get("status") == "success":
            total_entries += wiki_result.get("entries", 0)
    except Exception:
        results["wikipedia"] = {"entries": 0, "status": "error"}

    try:
        from forum_crawler import crawl_single_forum_topic
        forum_result = crawl_single_forum_topic(
            topic, [], max_per_source=5,
            sources=["stackexchange", "reddit", "hackernews", "devto"],
        )
        results["forums"] = {
            "entries": forum_result.get("entries", 0),
            "status": forum_result.get("status", "error"),
        }
        if forum_result.get("status") == "success":
            total_entries += forum_result.get("entries", 0)
    except Exception:
        results["forums"] = {"entries": 0, "status": "error"}

    # Crawl specific URLs if provided
    if urls:
        from site_crawler import add_site_job
        for url in urls[:MAX_RESEARCH_URLS]:
            try:
                r = add_site_job(url=url, topic=topic, max_pages=10, max_depth=1)
                url_entries = r.get("entries", 0)
                results[f"site:{url[:50]}"] = {
                    "entries": url_entries,
                    "status": r.get("status", "error"),
                }
                total_entries += url_entries
            except Exception:
                results[f"site:{url[:50]}"] = {"entries": 0, "status": "error"}

    # Auto-add topic to crawler configs for continuous learning
    try:
        from crawler import load_config, save_config
        config = load_config()
        topics = config.get("topics", [])
        if not any(t["name"].lower() == topic.lower() for t in topics):
            topics.append({
                "name": topic, "keywords": [], "enabled": True, "max_articles": 5,
            })
            config["topics"] = topics
            save_config(config)
    except Exception:
        pass

    # Trigger background retrain if we got new data
    if total_entries > 0:
        _background_retrain()

    return {
        "total_entries": total_entries,
        "sources": results,
        "topic": topic,
    }


def _background_retrain() -> None:
    """Retrain the knowledge model in a background thread, then reload."""
    def _do_retrain():
        try:
            from train_knowledge import train as _train_knowledge
            _train_knowledge(
                activation="tanh", optimizer="adam",
                lr=0.01, epochs=3000, hidden=256,
                augment=True, verbose=False,
            )
            _load_knowledge_model()
            print("Background retrain complete — knowledge model reloaded.")
        except Exception as exc:
            print(f"Background retrain failed: {type(exc).__name__}: {exc}")

    t = threading.Thread(target=_do_retrain, daemon=True)
    t.start()

@app.route("/chat", methods=["POST"])
def chat():
    """Answer a user question using the knowledge classifier.

    Also detects special commands:
      - "research <topic>" / "learn about <topic>" → multi-source crawl
      - URLs in the message → crawl the site(s) for knowledge
    """
    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question too long (max 2000 characters)"}), 400

    # Detect research / URL crawl commands
    cmd = _detect_chat_command(question)
    if cmd is not None:
        cmd_type = cmd["type"]
        if cmd_type == "research":
            topic = cmd["topic"]
            result = _execute_research(topic)
            total = result["total_entries"]
            if total > 0:
                answer = (
                    f"Researching **{topic}** — found {total} new knowledge entries "
                    f"from {len(result['sources'])} sources. "
                    "The model is retraining in the background. "
                    "Ask me about this topic again in a moment!"
                )
            else:
                answer = (
                    f"I searched multiple sources for '{topic}' but couldn't find "
                    "relevant knowledge entries. Try a more specific topic or "
                    "provide a URL to crawl."
                )
            return jsonify({
                "answer": answer,
                "confidence": 1.0,
                "domain": "research",
                "suggestions": [],
                "research": result,
            })

        elif cmd_type == "crawl":
            urls = cmd["urls"]
            topic = cmd["topic"]
            result = _execute_research(topic, urls=urls)
            total = result["total_entries"]
            url_list = ", ".join(urls[:3])
            if total > 0:
                answer = (
                    f"Crawling {url_list} — found {total} new entries. "
                    "The model is retraining in the background. "
                    "Ask me about this topic again shortly!"
                )
            else:
                answer = (
                    f"I crawled {url_list} but couldn't extract relevant knowledge. "
                    "The page might not have topic-relevant content, or it may be "
                    "behind authentication."
                )
            return jsonify({
                "answer": answer,
                "confidence": 1.0,
                "domain": "crawl",
                "suggestions": [],
                "research": result,
            })

    # Regular Q&A — requires loaded model
    if knowledge_model is None or knowledge_bow is None:
        return jsonify({"error": "Knowledge model not loaded. Train it first."}), 503

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
        answer = (
            "I'm not sure about that. Try asking about networking, internet, "
            "intranet, or security topics. You can also say "
            "'research <topic>' to teach me something new!"
        )
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


@app.route("/chat/research", methods=["POST"])
def chat_research():
    """Dedicated endpoint: research a topic across all sources.

    POST {"topic": "...", "urls": ["..."]}  →  multi-source crawl + background retrain.
    """
    data = request.get_json(force=True)
    topic = str(data.get("topic", "")).strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    if len(topic) > 200:
        return jsonify({"error": "Topic too long (max 200 characters)"}), 400

    urls = data.get("urls", [])
    if not isinstance(urls, list):
        urls = []
    # Validate URLs
    safe_urls = [u for u in urls[:MAX_RESEARCH_URLS] if isinstance(u, str) and _URL_PATTERN.match(u)]

    result = _execute_research(topic, urls=safe_urls if safe_urls else None)
    return jsonify({
        "status": "success" if result["total_entries"] > 0 else "no_results",
        "topic": topic,
        "total_entries": result["total_entries"],
        "sources": result["sources"],
        "message": (
            f"Found {result['total_entries']} entries about '{topic}'. "
            "Model retraining in background."
        ) if result["total_entries"] > 0 else f"No results found for '{topic}'.",
    })


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


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "knowledge_loaded": knowledge_model is not None,
        "datasets": list(TARGETS.keys()),
        "models_ready": list(models.keys()),
    })


# ── Routes: Brain & Watcher API ──────────────────────────────────────


@app.route("/brain/status", methods=["GET"])
def brain_status():
    """Return the LIBAIXBrain status summary."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_status())
    except Exception:
        return jsonify({"error": "Failed to retrieve brain status"}), 500


@app.route("/brain/briefing", methods=["GET"])
def brain_briefing():
    """Return a full session briefing from LIBAIXBrain."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_briefing())
    except Exception:
        return jsonify({"error": "Failed to build session briefing"}), 500


@app.route("/brain/scan", methods=["POST"])
def brain_scan():
    """Run a full brain scan cycle (scan → analyse → score → plan)."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_scan())
    except Exception:
        return jsonify({"error": "Failed to run brain scan cycle"}), 500


@app.route("/watcher/context", methods=["GET"])
def watcher_context():
    """Return the ML watcher's instant project context."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_context())
    except Exception:
        return jsonify({"error": "Failed to build watcher context"}), 500


@app.route("/watcher/cycle", methods=["POST"])
def watcher_cycle():
    """Run a full watcher monitoring cycle."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_cycle())
    except Exception:
        return jsonify({"error": "Failed to run watcher cycle"}), 500


# ── Routes: Extended Brain API ───────────────────────────────────────


@app.route("/brain/gaps", methods=["GET"])
def brain_gaps():
    """Return gap analysis results from the brain."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify({"gaps": _brain_analyse_gaps()})
    except Exception:
        return jsonify({"error": "Failed to analyse gaps"}), 500


@app.route("/brain/tasks", methods=["GET"])
def brain_tasks():
    """Return pending tasks from the brain's task queue."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        agent = request.args.get("agent")
        return jsonify({"tasks": _brain_pending_tasks(agent=agent)})
    except Exception:
        return jsonify({"error": "Failed to retrieve tasks"}), 500


@app.route("/brain/health", methods=["GET"])
def brain_health_score():
    """Return the project health score breakdown."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_health_score())
    except Exception:
        return jsonify({"error": "Failed to calculate health score"}), 500


@app.route("/brain/dependencies", methods=["GET"])
def brain_dependencies():
    """Return the module dependency graph."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_deps())
    except Exception:
        return jsonify({"error": "Failed to build dependency graph"}), 500


@app.route("/brain/complexity", methods=["GET"])
def brain_complexity():
    """Return module complexity scores."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_complexity())
    except Exception:
        return jsonify({"error": "Failed to score complexity"}), 500


@app.route("/brain/quality", methods=["GET"])
def brain_quality():
    """Return code quality metrics."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_quality())
    except Exception:
        return jsonify({"error": "Failed to measure code quality"}), 500


@app.route("/brain/knowledge-gaps", methods=["GET"])
def brain_knowledge_gaps():
    """Return knowledge gap recommendations."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_knowledge_gaps())
    except Exception:
        return jsonify({"error": "Failed to analyse knowledge gaps"}), 500


@app.route("/brain/impact/<module_name>", methods=["GET"])
def brain_impact(module_name):
    """Analyse impact of changing a specific module."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_impact(module_name))
    except Exception:
        return jsonify({"error": "Failed to analyse impact"}), 500


@app.route("/brain/stale", methods=["GET"])
def brain_stale():
    """Detect stale data files."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        max_days = request.args.get("days", 30, type=int)
        return jsonify(_brain_stale(max_age_days=max_days))
    except Exception:
        return jsonify({"error": "Failed to detect stale data"}), 500


@app.route("/brain/module/<module_name>", methods=["GET"])
def brain_module_summary(module_name):
    """Return a compact summary of a specific module."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Brain module not available"}), 503
    try:
        return jsonify(_brain_module_summary(module_name))
    except Exception:
        return jsonify({"error": "Failed to summarize module"}), 500


# ── Routes: Extended Watcher API ─────────────────────────────────────


@app.route("/watcher/growth", methods=["GET"])
def watcher_growth():
    """Track and return knowledge growth metrics."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_growth())
    except Exception:
        return jsonify({"error": "Failed to track knowledge growth"}), 500


@app.route("/watcher/config-drift", methods=["GET"])
def watcher_config_drift():
    """Detect configuration drift from baseline."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_config_drift())
    except Exception:
        return jsonify({"error": "Failed to detect config drift"}), 500


@app.route("/watcher/disk", methods=["GET"])
def watcher_disk():
    """Return disk usage measurements for project directories."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_disk_usage())
    except Exception:
        return jsonify({"error": "Failed to measure disk usage"}), 500


@app.route("/watcher/alerts", methods=["GET"])
def watcher_alerts():
    """Return alert summary from the watcher."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_alert_summary())
    except Exception:
        return jsonify({"error": "Failed to get alert summary"}), 500


@app.route("/watcher/alerts/clear", methods=["POST"])
def watcher_clear_alerts():
    """Clear all acknowledged alerts."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_clear_alerts())
    except Exception:
        return jsonify({"error": "Failed to clear alerts"}), 500


@app.route("/watcher/health", methods=["GET"])
def watcher_health():
    """Run and return a watcher health check."""
    if not _BRAIN_AVAILABLE:
        return jsonify({"error": "Watcher module not available"}), 503
    try:
        return jsonify(_watcher_health_check())
    except Exception:
        return jsonify({"error": "Failed to run health check"}), 500


# ── Routes: Boil Engine API ──────────────────────────────────────────


@app.route("/boil/status", methods=["GET"])
def boil_status():
    """Return boil engine status and state."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        return jsonify({
            "boiling": is_boiling(),
            "state": get_boil_state(),
            "config": load_boil_config(),
        })
    except Exception:
        return jsonify({"error": "Failed to get boil status"}), 500


@app.route("/boil/start", methods=["POST"])
def boil_start():
    """Start the boil engine background process."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        ok = start_boil_background()
        return jsonify({"started": ok, "boiling": is_boiling()})
    except Exception:
        return jsonify({"error": "Failed to start boil engine"}), 500


@app.route("/boil/stop", methods=["POST"])
def boil_stop():
    """Stop the boil engine background process."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        ok = stop_boil_background()
        return jsonify({"stopped": ok, "boiling": is_boiling()})
    except Exception:
        return jsonify({"error": "Failed to stop boil engine"}), 500


@app.route("/boil/tick", methods=["POST"])
def boil_tick():
    """Run a single boil improvement tick."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        result = run_boil_tick()
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Failed to run boil tick"}), 500


@app.route("/boil/log", methods=["GET"])
def boil_log():
    """Return recent improvement log entries."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        n = request.args.get("n", 50, type=int)
        return jsonify(get_improvement_log(min(n, 200)))
    except Exception:
        return jsonify({"error": "Failed to get boil log"}), 500


@app.route("/boil/config", methods=["GET", "POST"])
def boil_config_endpoint():
    """Get or update boil engine configuration."""
    if not _BOIL_AVAILABLE:
        return jsonify({"error": "Boil engine not available"}), 503
    try:
        if request.method == "POST":
            data = request.get_json(force=True)
            cfg = load_boil_config()
            cfg.update(data)
            save_boil_config(cfg)
            return jsonify({"saved": True, "config": cfg})
        return jsonify(load_boil_config())
    except Exception:
        return jsonify({"error": "Failed to handle boil config"}), 500


# ── Routes: Reasoning Engine API ─────────────────────────────────────


@app.route("/reason", methods=["POST"])
def reason_endpoint():
    """Apply deductive reasoning to a question."""
    if not _REASONING_AVAILABLE:
        return jsonify({"error": "Reasoning engine not available"}), 503
    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        result = reason_about(question)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Reasoning failed"}), 500


@app.route("/reason/stats", methods=["GET"])
def reason_stats():
    """Return reasoning engine statistics."""
    if not _REASONING_AVAILABLE:
        return jsonify({"error": "Reasoning engine not available"}), 503
    try:
        engine = get_reasoning_engine()
        return jsonify(engine.get_stats())
    except Exception:
        return jsonify({"error": "Failed to get reasoning stats"}), 500


@app.route("/reason/rebuild", methods=["POST"])
def reason_rebuild():
    """Rebuild the reasoning knowledge base."""
    if not _REASONING_AVAILABLE:
        return jsonify({"error": "Reasoning engine not available"}), 503
    try:
        result = _build_reasoning()
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Failed to rebuild reasoning base"}), 500


# ── Routes: Anonymous Crawler API ────────────────────────────────────


@app.route("/anon/stats", methods=["GET"])
def anon_stats():
    """Return anonymous crawler statistics."""
    if not _ANON_AVAILABLE:
        return jsonify({"error": "Anonymous crawler not available"}), 503
    try:
        return jsonify(get_anon_stats())
    except Exception:
        return jsonify({"error": "Failed to get anon stats"}), 500


@app.route("/anon/crawl", methods=["POST"])
def anon_crawl():
    """Anonymously crawl a URL for knowledge."""
    if not _ANON_AVAILABLE:
        return jsonify({"error": "Anonymous crawler not available"}), 503
    data = request.get_json(force=True)
    url = str(data.get("url", "")).strip()
    topic = str(data.get("topic", "general")).strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        result = anon_crawl_page(url, topic)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Anonymous crawl failed"}), 500


@app.route("/anon/crawl-site", methods=["POST"])
def anon_crawl_site_endpoint():
    """Anonymously crawl a full site for knowledge."""
    if not _ANON_AVAILABLE:
        return jsonify({"error": "Anonymous crawler not available"}), 503
    data = request.get_json(force=True)
    url = str(data.get("url", "")).strip()
    topic = str(data.get("topic", "general")).strip()
    max_pages = min(int(data.get("max_pages", 20)), 100)
    max_depth = min(int(data.get("max_depth", 2)), 5)
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        result = anon_crawl_site(url, topic, max_pages=max_pages, max_depth=max_depth)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Anonymous site crawl failed"}), 500


@app.route("/anon/config", methods=["GET", "POST"])
def anon_config_endpoint():
    """Get or update anonymous crawler configuration."""
    if not _ANON_AVAILABLE:
        return jsonify({"error": "Anonymous crawler not available"}), 503
    try:
        if request.method == "POST":
            data = request.get_json(force=True)
            cfg = load_anon_config()
            cfg.update(data)
            save_anon_config(cfg)
            return jsonify({"saved": True, "config": cfg})
        return jsonify(load_anon_config())
    except Exception:
        return jsonify({"error": "Failed to handle anon config"}), 500


# ── Routes: Form Filler API ──────────────────────────────────────────

# Form filler (lazy import)
try:
    from form_filler import (
        extract_forms,
        classify_field,
        parse_fill_prompt,
        fill_form,
        submit_form,
        save_profile,
        load_profiles,
        delete_profile,
        save_form_template,
        load_form_templates,
        get_fill_history,
        extract_validation_rules,
        FillProfile,
    )
    _FORM_AVAILABLE = True
except ImportError:
    _FORM_AVAILABLE = False


@app.route("/forms/extract", methods=["POST"])
def forms_extract():
    """Extract forms from a URL or HTML content."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    data = request.get_json(force=True)
    html = str(data.get("html", ""))
    url = str(data.get("url", "")).strip()

    # If URL provided, fetch the page first
    if url and not html:
        if _ANON_AVAILABLE:
            try:
                resp = anon_crawl_page(url, "form-extraction")
                html = resp.get("html", resp.get("text", ""))
            except Exception:
                return jsonify({"error": "Failed to fetch URL"}), 500
        else:
            return jsonify({"error": "Provide HTML content or enable anon crawler"}), 400

    if not html:
        return jsonify({"error": "HTML content or URL is required"}), 400

    try:
        forms = extract_forms(html, base_url=url)
        result = []
        for f in forms:
            fields = []
            for fld in f.fields:
                fields.append({
                    "name": fld.name, "type": fld.type, "id": fld.id,
                    "label": fld.label, "required": fld.required,
                    "options": fld.options, "value": fld.value,
                    "placeholder": fld.placeholder, "pattern": fld.pattern,
                    "semantic_type": classify_field(fld),
                })
            result.append({
                "url": f.url, "method": f.method, "action": f.action,
                "encoding": f.encoding, "csrf_field": f.csrf_field,
                "fields": fields,
            })
        return jsonify({"forms": result, "count": len(result)})
    except Exception:
        return jsonify({"error": "Failed to extract forms"}), 500


@app.route("/forms/fill", methods=["POST"])
def forms_fill():
    """Fill a form with provided values or profile."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    data = request.get_json(force=True)
    prompt = str(data.get("prompt", "")).strip()
    values = data.get("values", {})
    profile_name = data.get("profile", "")

    # Parse prompt if provided
    if prompt and not values:
        values = parse_fill_prompt(prompt)

    if not values:
        return jsonify({"error": "Provide values or a fill prompt"}), 400

    return jsonify({"parsed_values": values, "status": "ready"})


@app.route("/forms/profiles", methods=["GET", "POST"])
def forms_profiles():
    """List or create form fill profiles."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    try:
        if request.method == "POST":
            data = request.get_json(force=True)
            name = str(data.get("name", "")).strip()
            mappings = data.get("mappings", {})
            if not name or not mappings:
                return jsonify({"error": "Name and mappings required"}), 400
            from datetime import datetime, timezone
            profile = FillProfile(
                name=name,
                field_mappings=mappings,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            save_profile(profile)
            return jsonify({"saved": True, "name": name})
        profiles = load_profiles()
        return jsonify([{"name": p.name, "fields": len(p.field_mappings), "created_at": p.created_at} for p in profiles])
    except Exception:
        return jsonify({"error": "Failed to handle profiles"}), 500


@app.route("/forms/profiles/<name>", methods=["DELETE"])
def forms_delete_profile(name: str):
    """Delete a form fill profile."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    try:
        ok = delete_profile(name)
        return jsonify({"deleted": ok})
    except Exception:
        return jsonify({"error": "Failed to delete profile"}), 500


@app.route("/forms/templates", methods=["GET"])
def forms_templates():
    """List saved form templates."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    try:
        return jsonify(load_form_templates())
    except Exception:
        return jsonify({"error": "Failed to load form templates"}), 500


@app.route("/forms/history", methods=["GET"])
def forms_history():
    """Return form fill history."""
    if not _FORM_AVAILABLE:
        return jsonify({"error": "Form filler not available"}), 503
    try:
        n = request.args.get("n", 50, type=int)
        return jsonify(get_fill_history(min(n, 200)))
    except Exception:
        return jsonify({"error": "Failed to get form history"}), 500


# ── Routes: Consolidated Stats API ───────────────────────────────────


@app.route("/stats/all", methods=["GET"])
def stats_all():
    """Consolidated stats endpoint for the Stats page."""
    stats = {
        "knowledge": {
            "loaded": knowledge_model is not None,
            "total_answers": len(knowledge_answer_map),
            "domains": knowledge_domains,
            "vocab_size": knowledge_bow.vocab_size if knowledge_bow else 0,
        },
        "models": {
            "trained": list(models.keys()),
            "loss_history": {k: len(v) for k, v in loss_history.items()},
        },
        "boil": None,
        "reasoning": None,
        "anon": None,
        "brain": None,
        "memory": None,
    }
    if _BOIL_AVAILABLE:
        try:
            stats["boil"] = {
                "boiling": is_boiling(),
                "state": get_boil_state(),
            }
        except Exception:
            pass
    if _REASONING_AVAILABLE:
        try:
            stats["reasoning"] = get_reasoning_engine().get_stats()
        except Exception:
            pass
    if _ANON_AVAILABLE:
        try:
            stats["anon"] = get_anon_stats()
        except Exception:
            pass
    if _BRAIN_AVAILABLE:
        try:
            stats["brain"] = _brain_status()
        except Exception:
            pass
    try:
        stats["memory"] = build_startup_context()
    except Exception:
        pass
    return jsonify(stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
