#!/usr/bin/env python3
"""
train_knowledge.py — Train a neural network to classify and answer text questions.

Builds a classifier that maps user queries to knowledge entries using
bag-of-words vectorization and a softmax output neural network.

Run:
    python train_knowledge.py                    # Train with defaults
    python train_knowledge.py --epochs 5000      # Custom epochs
    python train_knowledge.py --activation tanh   # Custom activation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from knowledge_base import KNOWLEDGE, get_domains
from neural_network import NeuralNetwork
from vectorizer import BagOfWords

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "knowledge.npz"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.json"
ANSWER_MAP_PATH = MODEL_DIR / "answer_map.json"


def build_training_data(
    knowledge: list[tuple[str, str, str]],
) -> tuple[list[str], np.ndarray, list[str], dict[int, str]]:
    """Create training texts and one-hot label matrix from knowledge entries.

    Returns (questions, labels_onehot, domain_list, answer_map)
    where answer_map maps class index -> answer text.
    """
    questions = [q for q, _, _ in knowledge]
    answers = [a for _, a, _ in knowledge]
    n_classes = len(questions)

    # Each question is its own class (1-to-1 with an answer)
    labels = np.eye(n_classes, dtype=np.float64)

    # Also map index -> answer so we can retrieve it at inference
    answer_map = {i: answers[i] for i in range(n_classes)}

    domains = get_domains()
    return questions, labels, domains, answer_map


def augment_questions(knowledge: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Light data augmentation: add rephrased variants for each entry."""
    augmented = list(knowledge)
    for question, answer, domain in knowledge:
        # Variant 1: remove question mark, lowercase
        q2 = question.rstrip("?").strip()
        augmented.append((q2, answer, domain))

        # Variant 2: prepend "tell me about" or "explain"
        words = question.lower().replace("?", "").strip().split()
        # Remove leading "what is", "what are" etc.
        short = " ".join(words)
        for prefix in ("what is ", "what are ", "what is a ", "what is an ", "what is the "):
            if short.startswith(prefix):
                topic = short[len(prefix):]
                augmented.append((f"explain {topic}", answer, domain))
                augmented.append((f"tell me about {topic}", answer, domain))
                augmented.append((topic, answer, domain))
                break
    return augmented


def dedupe_entries(
    entries: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Drop exact duplicate (question, answer) pairs, preserving first-seen order.

    The crawlers re-fetch the same articles on every cycle, so the raw corpus
    is overwhelmingly exact duplicates. Collapsing them keeps every unique
    question/answer while keeping the training matrices a sane size.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for q, a, d in entries:
        key = (q, a)
        if key not in seen:
            seen.add(key)
            out.append((q, a, d))
    return out


def train(
    activation: str = "tanh",
    optimizer: str = "adam",
    lr: float = 0.01,
    epochs: int = 3000,
    hidden: int = 128,
    seed: int = 42,
    augment: bool = True,
    verbose: bool = True,
    early_stopping: bool = False,
    patience: int = 300,
    dropout_rate: float = 0.0,
    grad_clip: float | None = None,
    lr_schedule: str | None = None,
    val_split: float = 0.0,
    max_label_gb: float | None = None,
    batch_size: int | None = None,
) -> tuple[NeuralNetwork, BagOfWords, dict[int, str]]:
    """Train the knowledge classifier.  Returns (model, vectorizer, answer_map)."""

    # Prepare data — built-in + extra knowledge from crawler / uploads
    base = list(KNOWLEDGE)
    extra_dir = Path("data/extra_knowledge")
    if extra_dir.exists():
        from knowledge_base import load_extra_knowledge
        for fp in sorted(extra_dir.glob("*.json")):
            try:
                base.extend(load_extra_knowledge(fp))
            except Exception:
                pass

    # Collapse the ~99% duplicate corpus the crawlers produce before we build
    # any matrices.
    n_raw = len(base)
    base = dedupe_entries(base)
    if verbose and n_raw != len(base):
        print(f"Deduplicated {n_raw:,} entries → {len(base):,} unique (q,a) pairs")

    # A dense one-hot label matrix is (n_samples x n_classes) float64 and is the
    # dominant memory cost. Resolve a budget (env-overridable) and degrade
    # gracefully so we never attempt an impossible allocation: drop augmentation
    # first, then subsample as a last resort.
    if max_label_gb is None:
        max_label_gb = float(os.environ.get("LIBAIX_TRAIN_MAX_LABEL_GB", "2.0"))
    n_classes_est = len({a for _, a, _ in base})
    budget_bytes = max_label_gb * (1024 ** 3)

    data = augment_questions(base) if augment else base
    if augment and n_classes_est and len(data) * n_classes_est * 8 > budget_bytes:
        if verbose:
            print(
                f"One-hot labels would need {len(data) * n_classes_est * 8 / 1e9:.1f} GB "
                f"> {max_label_gb:.1f} GB budget — training without augmentation."
            )
        data = base
    if n_classes_est and len(data) * n_classes_est * 8 > budget_bytes:
        max_samples = max(1, int(budget_bytes // (n_classes_est * 8)))
        rng = np.random.RandomState(seed)
        keep = sorted(rng.permutation(len(data))[:max_samples].tolist())
        data = [data[i] for i in keep]
        if verbose:
            print(
                f"Subsampling to {len(data):,} samples to fit the "
                f"{max_label_gb:.1f} GB label-matrix budget."
            )

    questions = [q for q, _, _ in data]
    answers_source = [a for _, a, _ in data]
    n_samples = len(questions)

    # Map each unique answer to a class index
    unique_answers: list[str] = []
    answer_to_idx: dict[str, int] = {}
    for a in answers_source:
        if a not in answer_to_idx:
            answer_to_idx[a] = len(unique_answers)
            unique_answers.append(a)
    n_classes = len(unique_answers)

    answer_map = {i: a for i, a in enumerate(unique_answers)}

    # Build label matrix (one-hot)
    labels = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, a in enumerate(answers_source):
        labels[i, answer_to_idx[a]] = 1.0

    if verbose:
        print(f"Training data: {n_samples} questions → {n_classes} answer classes")
        print(f"Domains: {', '.join(get_domains())}")

    # Vectorize (fit on ALL data so vocab covers everything)
    bow = BagOfWords()
    X = bow.fit_transform(questions)
    vocab_size = bow.vocab_size

    # Train / validation split
    X_train, Y_train = X, labels
    X_val, Y_val = None, None
    if 0.0 < val_split < 1.0:
        rng = np.random.RandomState(seed)
        n_val = max(1, int(n_samples * val_split))
        idx = rng.permutation(n_samples)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, Y_train = X[train_idx], labels[train_idx]
        X_val, Y_val = X[val_idx], labels[val_idx]
        if verbose:
            print(f"Validation split: {len(train_idx)} train / {len(val_idx)} val")

    if verbose:
        print(f"Vocabulary size: {vocab_size}")

    # Build network: scale depth with dataset size
    if n_classes > 100:
        layer_sizes = [vocab_size, hidden, hidden, hidden // 2, n_classes]
    else:
        layer_sizes = [vocab_size, hidden, hidden // 2, n_classes]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=lr,
        activation=activation,
        optimizer=optimizer,
        loss="cross_entropy",
        softmax_output=True,
        seed=seed,
        dropout_rate=dropout_rate,
        grad_clip=grad_clip,
        lr_schedule=lr_schedule,
        batch_size=batch_size,
    )

    if verbose:
        print(f"Network: {' → '.join(map(str, layer_sizes))}")
        print(f"Activation: {activation} | Optimizer: {optimizer} | LR: {lr}")
        extras = []
        if early_stopping:
            extras.append(f"early_stop(patience={patience})")
        if dropout_rate > 0:
            extras.append(f"dropout={dropout_rate}")
        if grad_clip is not None:
            extras.append(f"grad_clip={grad_clip}")
        if lr_schedule:
            extras.append(f"lr_schedule={lr_schedule}")
        if extras:
            print(f"Enhancements: {', '.join(extras)}")
        print(f"Training for {epochs} epochs…\n")

    log_every = max(1, epochs // 10)
    losses = nn.train(
        X_train, Y_train,
        epochs=epochs,
        log_every=log_every if verbose else 0,
        early_stopping=early_stopping,
        patience=patience,
        x_val=X_val,
        y_val=Y_val,
    )

    # Evaluate accuracy on full data
    preds = nn.predict(X)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    # Validation accuracy
    val_accuracy = None
    if X_val is not None:
        val_preds = nn.predict(X_val)
        val_pred_classes = np.argmax(val_preds, axis=1)
        val_true_classes = np.argmax(Y_val, axis=1)
        val_accuracy = float(np.mean(val_pred_classes == val_true_classes))

    if verbose:
        print(f"\nTraining accuracy: {accuracy:.1%}")
        if val_accuracy is not None:
            print(f"Validation accuracy: {val_accuracy:.1%}")
        print(f"Final loss: {losses[-1]:.6f}")
        if early_stopping and len(losses) < epochs:
            print(f"Stopped early at epoch {len(losses)}")

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    nn.save(MODEL_PATH)
    bow.save(VECTORIZER_PATH)
    ANSWER_MAP_PATH.write_text(
        json.dumps(answer_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if verbose:
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Vectorizer saved to {VECTORIZER_PATH}")
        print(f"Answer map saved to {ANSWER_MAP_PATH}")

    # Remember training result in project memory
    try:
        from project_memory import remember_training_result
        remember_training_result(
            accuracy=float(accuracy),
            entries=n_samples,
            domains=len(get_domains()),
            config={
                "activation": activation,
                "optimizer": optimizer,
                "lr": lr,
                "epochs": len(losses),
                "hidden": hidden,
                "dropout_rate": dropout_rate,
                "grad_clip": grad_clip,
                "lr_schedule": lr_schedule,
                "final_loss": float(losses[-1]),
            },
        )
    except Exception:
        pass  # memory system is optional; training result already saved to disk

    return nn, bow, answer_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Train libaix knowledge classifier")
    parser.add_argument("--activation", choices=["sigmoid", "tanh", "relu"], default="tanh")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0-1)")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm")
    parser.add_argument(
        "--lr-schedule", choices=["step", "cosine"], default=None,
        help="Learning rate schedule",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.0,
        help="Fraction of data to hold out for validation (0-1)",
    )
    parser.add_argument(
        "--max-label-gb", type=float, default=None,
        help="Max GB for the one-hot label matrix before dropping augmentation "
        "and then subsampling (default 2.0, or $LIBAIX_TRAIN_MAX_LABEL_GB)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size for SGD (default: full-batch). Mini-batches "
        "converge in far fewer epochs on large datasets.",
    )
    args = parser.parse_args()

    train(
        activation=args.activation,
        optimizer=args.optimizer,
        lr=args.lr,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
        augment=not args.no_augment,
        early_stopping=args.early_stopping,
        patience=args.patience,
        dropout_rate=args.dropout,
        grad_clip=args.grad_clip,
        lr_schedule=args.lr_schedule,
        val_split=args.val_split,
        max_label_gb=args.max_label_gb,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
