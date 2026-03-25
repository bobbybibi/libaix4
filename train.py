#!/usr/bin/env python3
"""
train.py — Train a neural network on logic-gate problems and display results.

Run:
    python train.py                  # XOR (default)
    python train.py --dataset and    # AND gate
    python train.py --dataset or     # OR gate
    python train.py --dataset nand   # NAND gate
    python train.py --dataset all    # All gates
    python train.py --activation tanh --optimizer adam
"""

import argparse

import numpy as np

from neural_network import NeuralNetwork

# ── Datasets ──────────────────────────────────────────────────────────
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)

DATASETS: dict[str, np.ndarray] = {
    "xor":  np.array([[0], [1], [1], [0]], dtype=np.float64),
    "and":  np.array([[0], [0], [0], [1]], dtype=np.float64),
    "or":   np.array([[0], [1], [1], [1]], dtype=np.float64),
    "nand": np.array([[1], [1], [1], [0]], dtype=np.float64),
}


def train_and_eval(
    name: str,
    y: np.ndarray,
    activation: str,
    optimizer: str,
    lr: float,
    epochs: int,
    save_path: str | None,
) -> bool:
    nn = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        learning_rate=lr,
        activation=activation,
        optimizer=optimizer,
        seed=42,
    )

    print("=" * 55)
    print(f"  {name.upper()} Gate — activation={activation}  optimizer={optimizer}")
    print("=" * 55)

    losses = nn.train(INPUTS, y, epochs=epochs, log_every=max(1, epochs // 10))

    print(f"\n{'─' * 55}")
    predictions = nn.predict(INPUTS)
    all_correct = True
    for inp, target, pred in zip(INPUTS, y, predictions):
        rounded = int(round(float(pred[0])))
        correct = rounded == int(target[0])
        if not correct:
            all_correct = False
        mark = "✓" if correct else "✗"
        print(
            f"  Input: {inp}  |  Target: {int(target[0])}  "
            f"|  Pred: {pred[0]:.4f} → {rounded}  {mark}"
        )

    print()
    if all_correct:
        print(f"🎉  {name.upper()}: All outputs correct!")
    else:
        print(f"⚠️   {name.upper()}: Some outputs wrong — try more epochs or tweak hypers.")

    print(f"Final loss: {losses[-1]:.6f}\n")

    if save_path:
        nn.save(save_path)
        print(f"Model saved to {save_path}\n")

    return all_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Train libaix on logic-gate datasets")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="xor")
    parser.add_argument("--activation", choices=["sigmoid", "tanh", "relu"], default="sigmoid")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--save", type=str, default=None, help="Path to save .npz model")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = {}
    for name in names:
        ok = train_and_eval(name, DATASETS[name], args.activation, args.optimizer, args.lr, args.epochs, args.save)
        results[name] = ok

    if len(results) > 1:
        print("=" * 55)
        print("  Summary")
        print("=" * 55)
        for n, ok in results.items():
            print(f"  {n.upper():5s}  {'✓ passed' if ok else '✗ failed'}")
        print()


if __name__ == "__main__":
    main()
