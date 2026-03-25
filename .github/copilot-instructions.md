# libaix — Project Guidelines

## Architecture

- `neural_network.py` — Core `NeuralNetwork` class (forward/backward, activations, optimizers, save/load)
- `train.py` — CLI training script with argparse (multi-dataset, configurable hypers)
- `app.py` — Flask web UI with `/predict` and `/train` API endpoints
- `templates/index.html` — Single-page dark-themed playground (pure JS, no frameworks)
- `tests/test_neural_network.py` — pytest suite

## Code Style

- Python 3.10+, type hints on public APIs
- NumPy only for math — no external ML frameworks
- Lint with `ruff check`, format with `ruff format`

## Build and Test

```bash
make install     # pip install -r requirements.txt
make test        # pytest tests/ -v
make run         # Train XOR end-to-end
make lint        # ruff check
make check       # lint + test
python app.py    # Launch web UI on port 5000
```

## Conventions

- New features must include tests in `tests/test_neural_network.py`
- All datasets use the shared `INPUTS` array from `train.py`
- Web API endpoints return JSON; inputs are validated and clamped server-side
- Model config is stored as JSON inside `.npz` files (no pickle)
