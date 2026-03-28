# libaix — XOR Neural Network from Scratch

A minimal, dependency-light neural network that learns the **XOR** problem using only **NumPy** — no TensorFlow, PyTorch, or scikit-learn.

## Features

| Feature | Details |
|---|---|
| **Forward propagation** | Configurable multi-layer feed-forward network with sigmoid activation |
| **Back-propagation** | Full gradient descent with Xavier weight initialisation |
| **No ML frameworks** | Only NumPy for matrix math |
| **Single-command run** | `make run` installs deps, trains, and evaluates |
| **Tests** | pytest suite covering init, forward, backward, and XOR convergence |

## Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/lindapot-art/libaix.git
cd libaix

# Run everything (installs deps → trains → evaluates)
make run
```

### Or step-by-step

```bash
pip install -r requirements.txt
python train.py
```

## Project Assistant

An interactive bash assistant (`assist.sh`) wraps every common task into a single command so you can do minimal work:

```bash
./assist.sh help      # show all commands
./assist.sh setup     # install dependencies
./assist.sh train     # train the network
./assist.sh test      # run tests
./assist.sh lint      # lint with ruff
./assist.sh format    # auto-format code
./assist.sh check     # lint + tests
./assist.sh clean     # remove caches
./assist.sh all       # full pipeline: setup → lint → test → train
```

All the same targets are also available via `make`:

```bash
make help    # list targets
make all     # full pipeline
make check   # lint + tests
```

### CI / Automation

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push and PR to `main`:

- Lints with **ruff**
- Runs the **pytest** suite
- Smoke-tests training (`python train.py`)
- Tests across Python 3.10, 3.11, and 3.12

## Expected Output

```
==================================================
  XOR Neural Network — Training
==================================================
Epoch   1000  |  Loss: 0.178432
Epoch   2000  |  Loss: 0.021954
...
Epoch  10000  |  Loss: 0.000123

==================================================
  Results after training
==================================================
  Input: [0. 0.]  |  Target: 0  |  Prediction: 0.0102 → 0  ✓
  Input: [0. 1.]  |  Target: 1  |  Prediction: 0.9891 → 1  ✓
  Input: [1. 0.]  |  Target: 1  |  Prediction: 0.9889 → 1  ✓
  Input: [1. 1.]  |  Target: 0  |  Prediction: 0.0134 → 0  ✓

🎉  All XOR outputs learned correctly!
```

## Running Tests

```bash
make test
# or
python -m pytest tests/ -v
```

## Project Structure

```
libaix/
├── assist.sh                   # Project assistant (bash)
├── neural_network.py           # NeuralNetwork class (forward, backward, train, predict)
├── train.py                    # Training script — run this!
├── tests/
│   └── test_neural_network.py
├── requirements.txt            # numpy, pytest, ruff
├── Makefile                    # make run / make test / make lint / …
├── .github/workflows/ci.yml   # GitHub Actions CI
└── README.md
```

## How It Works

1. **Initialisation** — Weights are set via Xavier/Glorot initialisation; biases start at zero.
2. **Forward pass** — Input flows through each layer: `z = x·W + b`, then `a = σ(z)`.
3. **Loss** — Mean Squared Error between prediction and target.
4. **Backward pass** — Error is propagated layer-by-layer; weights and biases are updated proportional to the learning rate.
5. **Repeat** for 10 000 epochs until the network converges.

## License

MIT