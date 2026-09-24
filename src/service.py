"""
LLM Secret Detection Service
============================
Entry point — wires FastAPI app, middleware, and routers together.

All business logic lives in:
  analyzer.py      ← LLMAnalyzer (model + cache + versioning)
  security.py      ← SecurityManager, JWT, rate-limiting
  middleware.py    ← MonitoringMiddleware, SecurityHeadersMiddleware
  models.py        ← Pydantic request/response models
  routes/
    analyze.py     ← /analyze, /extension/analyze
    train.py       ← /train, /reset
    stats.py       ← /stats, /health, /cache/*, /database/*, /performance/*
    versioning.py  ← /versions/*, /ab/*
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .security import security_manager
from .analyzer import analyzer
from .database import db_manager
from .middleware import MonitoringMiddleware, SecurityHeadersMiddleware
from .streaming import add_streaming_routes
from .extension_auth import verify_extension_request

from .routes.analyze    import router as analyze_router
from .routes.train      import router as train_router
from .routes.stats      import router as stats_router
from .routes.versioning import router as versioning_router

try:
    from .performance_monitor import start_performance_monitoring
except ImportError:
    def start_performance_monitoring(): pass

# ── App ───────────────────────────────────────────────────────────────────────
_is_prod = os.getenv('ENVIRONMENT') == 'production'

app = FastAPI(
    title="DotAegis - AI Secret Detection & Security Engine",
    description="High-performance AI security engine for real-time secret detection and reputation intelligence",
    version="2.1.3",
    docs_url=None  if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
)

# ── Middleware (order matters — added in reverse execution order) ──────────────
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

_cors_origins = [o.strip() for o in
    os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080').split(',') if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins, allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"], max_age=86400,
)
app.add_middleware(MonitoringMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analyze_router)
app.include_router(train_router)
app.include_router(stats_router)
app.include_router(versioning_router)
# ── Streaming (SSE) routes ────────────────────────────────────────────────────
add_streaming_routes(app, analyzer, verify_extension_request)

# ── Root Landing ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "service": "DotAegis",
        "name": "DotAegis - AI Secret Detection & Security Engine",
        "status": "online",
        "version": "2.1.3",
        "health": "/health",
        "docs": "https://dotsuite.dev/ar/product/dotaegis"
    }

# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    db_manager.initialize()
    security_manager.load_api_keys()
    print(f"🔐 Security initialized — {len(security_manager.api_keys)} API key(s) loaded")
    print(f"⚡ Rate limit: {security_manager.max_requests_per_minute} req/min")
    try:
        start_performance_monitoring()
        print("📈 Performance monitoring started")
    except Exception as e:
        print(f"⚠️  Performance monitoring unavailable: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Shutting down — saving model...")
    analyzer.save_model()
    print("💾 Model saved")
