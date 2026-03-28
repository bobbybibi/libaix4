# libaix — Project Guidelines

## Architecture

- `neural_network.py` — Core `NeuralNetwork` class (forward/backward, activations, optimizers, softmax, cross-entropy, save/load)
- `vectorizer.py` — `BagOfWords` text vectorizer with TF-IDF (tokenize, fit, transform, save/load)
- `knowledge_base.py` — Curated Q&A knowledge entries for networking, internet, intranet, security
- `train.py` — CLI training script with argparse (multi-dataset, configurable hypers)
- `train_knowledge.py` — Knowledge classifier training pipeline (vectorize → train → save model)
- `app.py` — Flask web UI with AI chat (`/chat`), logic-gate playground (`/predict`, `/train`)
- `templates/index.html` — Tabbed UI: AI Chat + Playground (pure JS, no frameworks)
- `models/` — Saved model files (knowledge.npz, vectorizer.json, answer_map.json)
- `tests/` — pytest suite (test_neural_network.py, test_vectorizer.py, test_knowledge_base.py)

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
python train_knowledge.py  # Train AI knowledge model
python app.py    # Launch web UI on port 5000
```

## Conventions

- New features must include tests in `tests/`
- All datasets use the shared `INPUTS` array from `train.py`
- Web API endpoints return JSON; inputs are validated and clamped server-side
- Model config is stored as JSON inside `.npz` files (no pickle)
- Knowledge entries use (question, answer, domain) triples
