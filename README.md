# 🛡️ DotAegis

<p align="center">
  <img src="https://raw.githubusercontent.com/kareem2099/DotAegis/main/resources/banner.png" alt="DotAegis Banner" width="700" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>High-Throughput Neural Engine for Real-Time Secret Detection, Threat Intelligence & Anti-Poisoning</strong>
</p>

<p align="center">
  <a href="https://github.com/kareem2099/DotAegis/actions/workflows/test-and-deploy.yml"><img src="https://github.com/kareem2099/DotAegis/actions/workflows/test-and-deploy.yml/badge.svg" alt="CI/CD Status"/></a>
  <img src="https://img.shields.io/badge/Version-2.1.3-brightgreen.svg" alt="Version 2.1.3"/>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <a href="https://github.com/kareem2099/dotenvy"><img src="https://img.shields.io/badge/Ecosystem-DotSuite-00b4d8.svg" alt="DotSuite"/></a>
</p>

---

## 📖 Overview

**DotAegis** is an enterprise-grade AI microservice engineered to detect, classify, and neutralize API keys, tokens, database connection strings, and high-entropy credentials in real time. 

Built with **FastAPI**, **PyTorch/NumPy transformer architectures**, and a **35-dimension feature extractor**, DotAegis powers the intelligent backend of the [DotEnvy](https://github.com/kareem2099/dotenvy) VS Code extension and integrates seamlessly into CI/CD security pipelines.

---

## ⚡ Core Features

- 🧠 **35-Feature Neural Classifier**: Extracts Shannon entropy, bi-gram/tri-gram distribution, context risk signals, structural clues, and pattern heuristics.
- 🤝 **Dynamic Zero-Shared-Secret Handshake**: Ephemeral per-device registration storing unique client HMAC credentials securely inside the client's OS Keychain (`SecretStorage`).
- 🛡️ **Community Blacklist & Anti-Poisoning**: Consensus-driven hash blacklist with reputation scoring, preventing malicious poisoning attempts.
- ⚡ **Two-Tier Smart Caching**: Ultra-fast in-memory L1 LRU cache coupled with L2 Redis caching for sub-millisecond repeated analysis.
- 📡 **Server-Sent Events (SSE) Streaming**: Progressive real-time confidence streaming (`/extension/analyze/stream`) across 5 inspection stages.
- 🔒 **Defense-in-Depth Security**: Constant-time HMAC verification (`hmac.compare_digest`), sliding timestamp replay defense (5-min window), and strict rate limiters.

---

## 🏛️ 4-Layer Inspection Pipeline

DotAegis executes a layered filter pipeline that processes credentials at maximum speed with zero wasted compute:

```
[ Incoming Request / Keystroke ]
               │
               ▼
┌──────────────────────────────┐
│  L1: Instant Regex Gate      │  ──▶ Hit? (100% Confirmed Secret — 0ms latency)
└──────────────┬───────────────┘
               │ (Miss)
               ▼
┌──────────────────────────────┐
│  L2: Community Threat Cache  │  ──▶ Hit? (Known Leaked Hash — <1ms lookup)
└──────────────┬───────────────┘
               │ (Miss)
               ▼
┌──────────────────────────────┐
│  L3: Shannon Entropy Filter  │  ──▶ Shannon Entropy < 3.5? (Skip Neural Compute)
└──────────────┬───────────────┘
               │ (High Entropy)
               ▼
┌──────────────────────────────┐
│  L4: DotAegis Neural Model   │  ──▶ 35-Feature Transformer Analysis & Scoring
└──────────────────────────────┘
```

---

## 📁 Repository Structure

```
DotAegis/
├── Dockerfile                  # Multi-stage hardened production container
├── docker-compose.yml          # Full-stack orchestration (Service + Redis + Postgres + Nginx)
├── nginx.conf                  # Edge reverse proxy with security headers & rate limiting
├── railway.json                # Railway.app continuous deployment specification
├── requirements.txt            # Python dependencies
├── main.py                     # ASGI entrypoint for development & production
├── test_local.py               # Automated local/remote test suite (18 test scenarios)
├── train_model.py              # Neural model training & backpropagation pipeline
└── src/
    ├── service.py              # FastAPI app definition & middleware orchestration
    ├── analyzer.py             # LLMAnalyzer core orchestrator
    ├── model.py                # Custom neural network architecture & weights
    ├── attention.py            # Self-attention mechanism implementation
    ├── feature_extractor.py    # 35-feature extraction engine
    ├── extension_auth.py       # Per-device HMAC signature verification & handshake
    ├── security.py             # Internal API key authentication & sliding rate limiters
    ├── database.py             # SQLAlchemy models (SQLite fallback / PostgreSQL production)
    ├── cache_manager.py        # Two-tier cache manager (L1 LRU + L2 Redis)
    ├── streaming.py            # Server-Sent Events (SSE) streaming handler
    ├── performance_monitor.py  # Runtime memory & latency metrics collector
    └── routes/
        ├── analyze.py          # /analyze, /extension/analyze, /extension/register
        ├── stats.py            # /health, /stats, /metrics, /cache/*
        ├── train.py            # /train (human-in-the-loop continuous learning)
        └── versioning.py       # A/B model testing & version deployment
```

---

## 🚀 Quick Start

### 1. Local Development (Virtualenv)

```bash
# Clone the repository
git clone https://github.com/kareem2099/DotAegis.git
cd DotAegis

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Run development server with hot-reload
python main.py
```

The service will start on `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

---

### 2. Docker Compose (Full Stack)

Run DotAegis alongside PostgreSQL, Redis, and an Nginx reverse proxy:

```bash
docker-compose up -d
```

Check service health:
```bash
curl http://localhost:8000/health
```

---

### 3. Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Connect your GitHub repository to Railway.
2. Railway detects `railway.json` and the `Dockerfile` automatically.
3. Add environment variables in the **Variables** tab (refer to `.env.example`).
4. DotAegis deploys in seconds with automated SSL.

---

## ⚙️ Configuration & Environment Variables

| Variable | Description | Default | Required in Production |
|:---|:---|:---|:---:|
| `ENVIRONMENT` | Runtime mode (`development` or `production`) | `development` | Yes |
| `API_KEY` | Comma-separated API keys for admin/internal endpoints | `""` | Yes |
| `JWT_SECRET` | 256-bit secret used for internal cryptographic tokens | Generated | Yes |
| `DATABASE_URL` | SQLAlchemy connection string (PostgreSQL or SQLite) | `sqlite:///./llm_service.db` | Recommended |
| `REDIS_URL` | Redis connection URL for L2 distributed cache | `""` (L1 LRU fallback) | Optional |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Max requests per minute per IP address | `60` | No |
| `EXTENSION_RATE_LIMIT` | Max requests per minute per extension device | `30` | No |
| `REGISTRATION_RATE_LIMIT_PER_IP`| Max device handshakes allowed per hour per IP | `10` | No |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` | No |
| `PORT` | HTTP port to bind the server | `8000` | No |

---

## 📡 API Reference

### Extension Endpoints (Dynamic Handshake & HMAC Signed)

| Method | Endpoint | Description | Auth Required |
|:---|:---|:---|:---:|
| `POST` | `/extension/register` | Dynamic device registration handshake | IP Rate Limit (10/hr) |
| `POST` | `/extension/analyze` | High-confidence secret detection | HMAC Signature (`X-Extension-*`) |
| `POST` | `/extension/analyze/stream` | Real-time SSE 5-stage analysis streaming | HMAC Signature (`X-Extension-*`) |
| `POST` | `/extension/feedback` | User confirmation/FP training samples | HMAC Signature (`X-Extension-*`) |
| `GET` | `/extension/blacklist` | Sync community threat blacklist | HMAC Signature (`X-Extension-*`) |
| `POST` | `/extension/blacklist/add` | Submit detected hash to staging queue | HMAC Signature (`X-Extension-*`) |
| `POST` | `/extension/blacklist/report_fp`| Report false-positive hash | HMAC Signature (`X-Extension-*`) |

### Service & Administrative Endpoints

| Method | Endpoint | Description | Auth Required |
|:---|:---|:---|:---:|
| `GET` | `/health` | Liveness & readiness probe | Public |
| `GET` | `/stats` | Service, model, and cache analytics | Public |
| `GET` | `/metrics` | Prometheus metrics endpoint | Public |
| `POST` | `/analyze` | Direct API secret analysis | `X-API-KEY` or Bearer Token |
| `POST` | `/train` | Continuous learning training step | `X-API-KEY` |
| `POST` | `/reset` | Reset model weights to baseline | `X-API-KEY` |
| `POST` | `/cache/clear` | Invalidate L1 & L2 cache stores | `X-API-KEY` |
| `POST` | `/database/cleanup` | Purge aged analytics data | `X-API-KEY` |

---

## 🧪 Testing

Run the automated test suite against a running local or staging instance:

```bash
# Run against local development server
python test_local.py --url http://localhost:8000

# Run model self-training & validation (15 epochs)
python train_model.py --local --epochs 15
```

---

## 🔗 The DotSuite Ecosystem

DotAegis is part of the **DotSuite** developer toolchain:

- **[DotEnvy](https://github.com/kareem2099/dotenvy)** — Intelligent `.env` & secret manager for VS Code.
- **[DotGhostBoard](https://github.com/kareem2099/DotGhostBoard)** — Secure clipboard & developer productivity hub.
- **[DotFetch](https://github.com/kareem2099/DotFetch)** — Fast, lightweight API exploration & testing tool.

---

## 📄 License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Crafted with precision by <strong><a href="https://github.com/kareem2099">Kareem Ehab</a></strong> &bull; <strong><a href="https://www.dotsuite.dev/en">DotSuite</a></strong>
</p>
