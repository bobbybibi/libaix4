# libaix — Self-Deploying AI Knowledge Engine

A drag-and-drop AI chatbot and neural network platform. Drop the folder on any PC or server, run one command, and it works. No manual setup.

## 🚀 Quick Start (Drag & Drop)

**Prerequisite:** Python 3.10+ installed ([download](https://www.python.org))

### On your PC (Windows, Mac, Linux)

```bash
# Option 1: Double-click
#   Windows → double-click start.bat
#   Mac/Linux → double-click start.sh (or run ./start.sh)

# Option 2: Command line
python start.py
```

That's it. The launcher automatically:
1. ✅ Installs all dependencies (numpy, flask, etc.)
2. ✅ Trains the knowledge AI model (first time only)
3. ✅ Starts the web server

Then open **http://localhost:5000** in your browser.

### Options

```bash
python start.py                  # Default (port 5000)
python start.py --port 8080      # Custom port
python start.py --host 127.0.0.1 # Localhost only (more secure)
python start.py --retrain        # Force retrain the AI model
```

### On Shared Hosting (cPanel / Passenger)

1. Upload the entire `libaix/` folder to your hosting account
2. In cPanel → **Setup Python App**:
   - Application root: `/home/<user>/libaix`
   - Startup file: `passenger_wsgi.py`
   - Entry point: `application`
3. Open a terminal (SSH or cPanel Terminal) and run:
   ```bash
   cd ~/libaix
   pip install -r requirements.txt
   python train_knowledge.py
   ```
4. Restart the Python app in cPanel → done!

Files included for hosting: `passenger_wsgi.py`, `.htaccess`

## Features

| Feature | Details |
|---|---|
| **AI Chat** | Knowledge Q&A chatbot (networking, security, internet, intranet, wifi) |
| **Admin Dashboard** | `/admin` — manage knowledge, crawlers, file uploads, ML engine |
| **Retrieval Engine** | TF-IDF cosine nearest-question retrieval (`retrieval.py`) — answers instantly with no training; preferred over the classifier when its index is present |
| **ML Self-Optimization** | Auto-optimizes hyperparameters, stabilizes training, prevents forgetting |
| **Knowledge Crawlers** | Wikipedia + forum crawlers (StackExchange, Reddit, HN) for auto-learning |
| **Logic Gate Playground** | Interactive XOR/AND/OR/NAND neural network trainer |
| **Local Scheduler** | Background automation for crawling, training, and ML growth cycles |
| **Self-Deploying** | `start.py` / `start.sh` / `start.bat` — zero-config launch |
| **No ML Frameworks** | Pure NumPy neural network — no TensorFlow/PyTorch needed |


## SaaS Productization Foundation

This repository now includes initial SaaS scaffolding for web + mobile clients:

- **Multi-tenant database layer** (`saas_db.py`) with tenants, users, subscriptions, analytics events
- **Versioned API** (`api_v1.py`) under `/api/v1` for auth, onboarding, plans, chat/research wrappers, analytics capture
- **Background job queue** (`job_queue.py`, `worker.py`) using Redis + RQ for research/retraining/retriever rebuild jobs
- **Container deployment** (`Dockerfile`, `docker-compose.yml`, `.env.example`) with web + worker + Postgres + Redis
- **Web onboarding UX** in `templates/index.html` for trade-pack selection and trade disclaimers

### API v1 quick endpoints

- `GET /api/v1/health`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/onboarding/select-trade`
- `GET /api/v1/billing/plans`
- `POST /api/v1/billing/subscribe`
- `POST /api/v1/chat`
- `POST /api/v1/research`
- `POST /api/v1/analytics/query-feedback`

### Docker quick start

```bash
docker compose up --build
```

Then open `http://localhost:5000`.

### FTP Website Deployment (photon-bounce)

A workflow is included at `.github/workflows/deploy-ftp.yml` to deploy `site/` to your FTP host and remote subfolder.
It automatically performs a backup of existing remote content before replacing it.

Required GitHub repository secrets:

- `FTP_HOST` (example: `ftp.photon-bounce.com`)
- `FTP_USER`
- `FTP_PASS`
- `FTP_REMOTE_DIR` (optional; defaults to `/libaix`)

The workflow runs on:

- manual dispatch (`workflow_dispatch`)
- pushes to `main` that modify `site/**`

## Project Assistant

```bash
./assist.sh help      # show all commands
./assist.sh setup     # install dependencies
./assist.sh train     # train the XOR network
./assist.sh test      # run tests
./assist.sh lint      # lint with ruff
./assist.sh check     # lint + tests
./assist.sh all       # full pipeline: setup → lint → test → train
```

Or via `make`:
```bash
make help    # list targets
make all     # full pipeline
make check   # lint + tests
```

## CI / Automation

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:
- Lints with **ruff**, runs **pytest**, smoke-tests training
- Tests across Python 3.10, 3.11, 3.12

Local scheduler (`python local_scheduler.py`) provides offline automation:
- Auto-trains knowledge model (single mini-batch config once daily, plus on-demand; mini-batch SGD converges in far fewer epochs)
- Crawls Wikipedia & forums for new knowledge
- Runs ML self-growth optimization cycles

## Project Structure

```
libaix/
├── start.py                  # ⭐ Self-deploying launcher (run this!)
├── start.sh                  # One-click launcher (Linux/Mac)
├── start.bat                 # One-click launcher (Windows)
├── app.py                    # Flask web server (chat, playground, API)
├── admin.py                  # Admin dashboard blueprint
├── neural_network.py         # Core neural network (forward/backward/train)
├── vectorizer.py             # Bag-of-words text vectorizer with TF-IDF
├── retrieval.py              # Zero-training retrieval engine (TF-IDF cosine lookup)
├── knowledge_base.py         # Curated Q&A knowledge entries
├── train_knowledge.py        # Knowledge classifier training pipeline
├── ml_engine.py              # ML self-optimization engine
├── crawler.py                # Wikipedia knowledge crawler
├── forum_crawler.py          # Forum crawler (StackExchange, Reddit, etc.)
├── local_scheduler.py        # Background job scheduler
├── passenger_wsgi.py         # WSGI entry for shared hosting
├── scripts/
│   └── build_retrieval_index.py  # Builds the retrieval index
├── .htaccess                 # Apache/Passenger config
├── models/                   # Trained model files
├── data/                     # Config, knowledge data, crawler output
├── templates/                # HTML templates (chat UI, admin dashboard)
├── tests/                    # pytest test suite
├── requirements.txt          # Python dependencies
├── Makefile                  # Build/test/lint targets
├── assist.sh                 # Project assistant (bash)
└── README.md
```

## How It Works

1. **Knowledge Base** — Curated Q&A triples (question, answer, domain) plus crawled knowledge
2. **Vectorization** — Bag-of-words with TF-IDF converts questions to numeric vectors
3. **Retrieval Engine** — `KnowledgeRetriever` (in `retrieval.py`) vectorizes all known questions using TF-IDF with L2-normalization and answers queries via cosine-similarity nearest-question lookup. The `/chat` endpoint prefers retrieval when an index is loaded and falls back to the neural classifier otherwise. `start.py` builds the index automatically on launch; it can also be (re)built with `python scripts/build_retrieval_index.py`.
4. **Neural Network** — Multi-layer softmax classifier maps vectors to answer classes
5. **ML Engine** — Self-assesses accuracy, auto-optimizes hyperparameters, prevents forgetting
6. **Web UI** — Flask serves the chat interface, admin dashboard, and logic-gate playground

## Running Tests

```bash
python -m pytest tests/ -v    # or: make test
```

## License

MIT