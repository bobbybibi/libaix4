"""
ml_engine.py — Machine Learning self-growth, stabilization, and optimization engine.

Provides automated mechanisms for the AI to:
  1. Self-assess model quality (accuracy, confidence, coverage)
  2. Auto-optimize hyperparameters (grid search on activation, lr, optimizer)
  3. Detect and fix knowledge gaps (low-confidence domains)
  4. Stabilize training (prevent catastrophic forgetting, track best models)
  5. Schedule growth cycles (continuous improvement loop)

All controllable from the admin dashboard.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ENGINE_CONFIG_PATH = Path("data/ml_engine_config.json")
ENGINE_HISTORY_PATH = Path("data/ml_engine_history.json")
MODEL_DIR = Path("models")
BACKUP_DIR = Path("models/backups")


# ── Config ───────────────────────────────────────────────────────────

def load_engine_config() -> dict:
    if ENGINE_CONFIG_PATH.exists():
        return json.loads(ENGINE_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = _default_config()
    save_engine_config(cfg)
    return cfg


def save_engine_config(config: dict) -> None:
    ENGINE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_config() -> dict:
    return {
        "auto_optimize": True,
        "auto_stabilize": True,
        "auto_grow": True,
        "min_accuracy_threshold": 0.90,
        "confidence_threshold": 0.60,
        "max_epochs_per_cycle": 10000,
        "optimization_configs": [
            {"activation": "tanh", "optimizer": "adam", "lr": 0.01, "epochs": 5000},
            {"activation": "tanh", "optimizer": "adam", "lr": 0.005, "epochs": 8000},
            {"activation": "relu", "optimizer": "adam", "lr": 0.005, "epochs": 5000},
            {"activation": "tanh", "optimizer": "adam", "lr": 0.002, "epochs": 15000},
            {"activation": "tanh", "optimizer": "adam", "lr": 0.001, "epochs": 20000},
        ],
        "growth_targets": {
            "min_entries": 200,
            "min_domains": 8,
            "target_accuracy": 0.98,
        },
        "last_optimization": None,
        "last_assessment": None,
        "cycle_count": 0,
    }


# ── History tracking ─────────────────────────────────────────────────

def load_history() -> list[dict]:
    if ENGINE_HISTORY_PATH.exists():
        return json.loads(ENGINE_HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def _save_history(history: list[dict]) -> None:
    ENGINE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep last 100 entries
    history = history[-100:]
    ENGINE_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _record_event(event_type: str, data: dict) -> None:
    history = load_history()
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **data,
    })
    _save_history(history)


# ── Self-Assessment ──────────────────────────────────────────────────

def assess_model() -> dict:
    """Evaluate current model quality: accuracy, confidence, domain coverage."""
    from knowledge_base import KNOWLEDGE, get_domains
    from neural_network import NeuralNetwork
    from vectorizer import BagOfWords

    model_path = MODEL_DIR / "knowledge.npz"
    vec_path = MODEL_DIR / "vectorizer.json"
    map_path = MODEL_DIR / "answer_map.json"

    if not all(p.exists() for p in [model_path, vec_path, map_path]):
        return {"status": "no_model", "message": "No trained model found"}

    model = NeuralNetwork.load(model_path)
    bow = BagOfWords.load(vec_path)
    answer_map = json.loads(map_path.read_text(encoding="utf-8"))

    # Load extra knowledge too
    all_knowledge = list(KNOWLEDGE)
    extra_dir = Path("data/extra_knowledge")
    if extra_dir.exists():
        from knowledge_base import load_extra_knowledge
        for fp in sorted(extra_dir.glob("*.json")):
            try:
                all_knowledge.extend(load_extra_knowledge(fp))
            except Exception:
                pass

    questions = [q for q, _, _ in all_knowledge]
    answers = [a for _, a, _ in all_knowledge]
    domains_list = [d for _, _, d in all_knowledge]

    # Predict
    X = bow.transform(questions)
    preds = model.predict(X)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    # Accuracy per domain
    domain_stats: dict[str, dict] = {}
    correct_total = 0

    for i, (q, ans, dom) in enumerate(all_knowledge):
        pred_answer = answer_map.get(str(pred_classes[i]), "")
        is_correct = pred_answer == ans
        if is_correct:
            correct_total += 1

        if dom not in domain_stats:
            domain_stats[dom] = {
                "total": 0, "correct": 0,
                "confidences": [], "low_confidence": [],
            }
        domain_stats[dom]["total"] += 1
        if is_correct:
            domain_stats[dom]["correct"] += 1
        domain_stats[dom]["confidences"].append(float(confidences[i]))
        if confidences[i] < 0.6:
            domain_stats[dom]["low_confidence"].append(q)

    overall_accuracy = correct_total / len(questions) if questions else 0
    avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0

    # Summarize domain stats
    domain_summary = {}
    weak_domains = []
    for dom, stats in domain_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        avg_conf = np.mean(stats["confidences"]) if stats["confidences"] else 0
        domain_summary[dom] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(acc, 4),
            "avg_confidence": round(float(avg_conf), 4),
            "low_confidence_count": len(stats["low_confidence"]),
        }
        if acc < 0.90 or avg_conf < 0.60:
            weak_domains.append(dom)

    result = {
        "status": "ok",
        "overall_accuracy": round(overall_accuracy, 4),
        "avg_confidence": round(avg_confidence, 4),
        "total_entries": len(questions),
        "total_classes": len(answer_map),
        "domains": domain_summary,
        "weak_domains": weak_domains,
        "needs_optimization": overall_accuracy < 0.95 or len(weak_domains) > 0,
        "assessed_at": datetime.now(timezone.utc).isoformat(),
    }

    config = load_engine_config()
    config["last_assessment"] = result["assessed_at"]
    save_engine_config(config)
    _record_event("assessment", {
        "accuracy": result["overall_accuracy"],
        "confidence": result["avg_confidence"],
        "weak_domains": weak_domains,
    })

    return result


# ── Auto-Optimize ────────────────────────────────────────────────────

def optimize_model(configs: list[dict] | None = None) -> dict:
    """Try multiple hyperparameter configs, keep the best."""
    from train_knowledge import train

    engine_config = load_engine_config()
    configs = configs or engine_config.get("optimization_configs", [])

    # Backup current model
    _backup_current_model()

    best_accuracy = 0.0
    best_config = None
    best_model = None
    best_bow = None
    best_answer_map = None
    results = []

    for i, cfg in enumerate(configs):
        try:
            model, bow, answer_map = train(
                activation=cfg.get("activation", "tanh"),
                optimizer=cfg.get("optimizer", "adam"),
                lr=cfg.get("lr", 0.01),
                epochs=cfg.get("epochs", 5000),
                hidden=cfg.get("hidden", 256),
                augment=True,
                verbose=False,
            )

            # Evaluate
            from knowledge_base import KNOWLEDGE
            questions = [q for q, _, _ in KNOWLEDGE]
            answers = [a for _, a, _ in KNOWLEDGE]
            X = bow.transform(questions)
            preds = model.predict(X)
            pred_classes = np.argmax(preds, axis=1)

            correct = sum(1 for j, q in enumerate(questions)
                          if answer_map.get(pred_classes[j], "") == answers[j])
            accuracy = correct / len(questions) if questions else 0

            results.append({
                "config": cfg,
                "accuracy": round(accuracy, 4),
                "status": "ok",
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = cfg
                best_model = model
                best_bow = bow
                best_answer_map = answer_map

        except Exception as e:
            results.append({
                "config": cfg,
                "accuracy": 0,
                "status": f"error: {e}",
            })

    # Save best model
    if best_model and best_accuracy > 0:
        MODEL_DIR.mkdir(exist_ok=True)
        best_model.save(MODEL_DIR / "knowledge.npz")
        best_bow.save(MODEL_DIR / "vectorizer.json")
        (MODEL_DIR / "answer_map.json").write_text(
            json.dumps(best_answer_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    engine_config["last_optimization"] = datetime.now(timezone.utc).isoformat()
    engine_config["cycle_count"] = engine_config.get("cycle_count", 0) + 1
    save_engine_config(engine_config)

    _record_event("optimization", {
        "best_accuracy": round(best_accuracy, 4),
        "best_config": best_config,
        "configs_tested": len(configs),
    })

    return {
        "status": "success",
        "best_accuracy": round(best_accuracy, 4),
        "best_config": best_config,
        "results": results,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Stabilization ───────────────────────────────────────────────────

def stabilize_model() -> dict:
    """
    Check model stability — if accuracy dropped, restore from backup.
    Prevents catastrophic forgetting.
    """
    assessment = assess_model()
    if assessment.get("status") != "ok":
        return {"status": "no_model", "action": "none"}

    config = load_engine_config()
    threshold = config.get("min_accuracy_threshold", 0.90)
    current_accuracy = assessment["overall_accuracy"]

    if current_accuracy >= threshold:
        _record_event("stabilization_check", {
            "accuracy": current_accuracy,
            "action": "stable",
        })
        return {
            "status": "stable",
            "accuracy": current_accuracy,
            "threshold": threshold,
            "action": "none",
            "message": f"Model is stable at {current_accuracy:.1%} (threshold: {threshold:.0%})",
        }

    # Try to restore from backup
    restored = _restore_best_backup()
    if restored:
        new_assessment = assess_model()
        _record_event("stabilization_restore", {
            "old_accuracy": current_accuracy,
            "new_accuracy": new_assessment["overall_accuracy"],
        })
        return {
            "status": "restored",
            "old_accuracy": current_accuracy,
            "new_accuracy": new_assessment["overall_accuracy"],
            "action": "restored_backup",
            "message": f"Restored backup model ({new_assessment['overall_accuracy']:.1%})",
        }

    # No backup available — force optimize
    opt_result = optimize_model()
    _record_event("stabilization_reoptimize", {
        "old_accuracy": current_accuracy,
        "new_accuracy": opt_result["best_accuracy"],
    })
    return {
        "status": "reoptimized",
        "old_accuracy": current_accuracy,
        "new_accuracy": opt_result["best_accuracy"],
        "action": "reoptimized",
        "message": f"Reoptimized from {current_accuracy:.1%} to {opt_result['best_accuracy']:.1%}",
    }


# ── Self-Growth ──────────────────────────────────────────────────────

def assess_growth() -> dict:
    """Check what the AI still needs to learn."""
    assessment = assess_model()
    if assessment.get("status") != "ok":
        return {"status": "no_model", "suggestions": []}

    config = load_engine_config()
    targets = config.get("growth_targets", {})

    suggestions = []

    # Check entry count
    min_entries = targets.get("min_entries", 200)
    if assessment["total_entries"] < min_entries:
        suggestions.append({
            "type": "need_more_data",
            "message": f"Only {assessment['total_entries']} entries, target is {min_entries}",
            "priority": "high",
        })

    # Check weak domains
    for dom in assessment.get("weak_domains", []):
        dom_stats = assessment["domains"].get(dom, {})
        suggestions.append({
            "type": "weak_domain",
            "domain": dom,
            "accuracy": dom_stats.get("accuracy", 0),
            "confidence": dom_stats.get("avg_confidence", 0),
            "message": f"Domain '{dom}' needs improvement (acc: {dom_stats.get('accuracy', 0):.0%})",
            "priority": "high",
        })

    # Check target accuracy
    target_acc = targets.get("target_accuracy", 0.98)
    if assessment["overall_accuracy"] < target_acc:
        suggestions.append({
            "type": "accuracy_gap",
            "current": assessment["overall_accuracy"],
            "target": target_acc,
            "message": f"Accuracy {assessment['overall_accuracy']:.1%} below target {target_acc:.0%}",
            "priority": "medium",
        })

    # Check domain count
    min_domains = targets.get("min_domains", 8)
    domain_count = len(assessment.get("domains", {}))
    if domain_count < min_domains:
        suggestions.append({
            "type": "need_more_domains",
            "current": domain_count,
            "target": min_domains,
            "message": f"Only {domain_count} domains, target is {min_domains}",
            "priority": "low",
        })

    return {
        "status": "ok",
        "accuracy": assessment["overall_accuracy"],
        "entries": assessment["total_entries"],
        "domains": len(assessment.get("domains", {})),
        "suggestions": suggestions,
        "needs_growth": len(suggestions) > 0,
    }


def run_growth_cycle() -> dict:
    """
    Full self-improvement cycle:
      1. Assess current model
      2. Stabilize if needed
      3. Optimize hyperparameters
      4. Report results
    """
    results = {"steps": []}

    # Step 1: Assess
    assessment = assess_model()
    results["steps"].append({
        "step": "assess",
        "accuracy": assessment.get("overall_accuracy", 0),
        "entries": assessment.get("total_entries", 0),
    })

    if assessment.get("status") != "ok":
        results["status"] = "no_model"
        return results

    config = load_engine_config()

    # Step 2: Stabilize if accuracy is low
    if config.get("auto_stabilize") and assessment.get("needs_optimization"):
        stab_result = stabilize_model()
        results["steps"].append({
            "step": "stabilize",
            "action": stab_result.get("action", "none"),
            "accuracy": stab_result.get("new_accuracy", stab_result.get("accuracy", 0)),
        })

    # Step 3: Optimize
    if config.get("auto_optimize"):
        opt_result = optimize_model()
        results["steps"].append({
            "step": "optimize",
            "best_accuracy": opt_result.get("best_accuracy", 0),
            "configs_tested": len(opt_result.get("results", [])),
        })

    # Step 4: Final assessment
    final = assess_model()
    results["steps"].append({
        "step": "final_assessment",
        "accuracy": final.get("overall_accuracy", 0),
        "confidence": final.get("avg_confidence", 0),
    })

    results["status"] = "success"
    results["final_accuracy"] = final.get("overall_accuracy", 0)
    results["improvement"] = round(
        final.get("overall_accuracy", 0) - assessment.get("overall_accuracy", 0), 4
    )
    results["ran_at"] = datetime.now(timezone.utc).isoformat()

    config["cycle_count"] = config.get("cycle_count", 0) + 1
    save_engine_config(config)
    _record_event("growth_cycle", {
        "start_accuracy": assessment.get("overall_accuracy", 0),
        "final_accuracy": final.get("overall_accuracy", 0),
    })

    return results


# ── Model backup / restore ───────────────────────────────────────────

def _backup_current_model() -> Path | None:
    """Backup current model files before optimization."""
    model_path = MODEL_DIR / "knowledge.npz"
    if not model_path.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_subdir = BACKUP_DIR / ts
    backup_subdir.mkdir(exist_ok=True)

    for fname in ["knowledge.npz", "vectorizer.json", "answer_map.json"]:
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, backup_subdir / fname)

    # Keep only last 5 backups
    backups = sorted(BACKUP_DIR.iterdir())
    while len(backups) > 5:
        oldest = backups.pop(0)
        if oldest.is_dir():
            shutil.rmtree(oldest)

    return backup_subdir


def _restore_best_backup() -> bool:
    """Restore the most recent backup."""
    if not BACKUP_DIR.exists():
        return False
    backups = sorted(BACKUP_DIR.iterdir())
    if not backups:
        return False

    latest = backups[-1]
    if not latest.is_dir():
        return False

    for fname in ["knowledge.npz", "vectorizer.json", "answer_map.json"]:
        src = latest / fname
        if src.exists():
            shutil.copy2(src, MODEL_DIR / fname)

    return True


# ── Dashboard stats ──────────────────────────────────────────────────

def get_engine_stats() -> dict:
    """Get full ML engine status for admin dashboard."""
    config = load_engine_config()
    history = load_history()

    # Extract accuracy trend from history
    accuracy_trend = []
    for event in history:
        if event.get("type") in ("assessment", "optimization", "growth_cycle"):
            acc = event.get("accuracy") or event.get("best_accuracy") or event.get("final_accuracy")
            if acc:
                accuracy_trend.append({
                    "timestamp": event.get("timestamp", ""),
                    "accuracy": acc,
                    "type": event["type"],
                })

    backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []

    return {
        "config": config,
        "history_count": len(history),
        "recent_events": history[-10:],
        "accuracy_trend": accuracy_trend[-20:],
        "backups": len(backups),
        "auto_optimize": config.get("auto_optimize", True),
        "auto_stabilize": config.get("auto_stabilize", True),
        "auto_grow": config.get("auto_grow", True),
        "cycle_count": config.get("cycle_count", 0),
        "last_optimization": config.get("last_optimization"),
        "last_assessment": config.get("last_assessment"),
    }
