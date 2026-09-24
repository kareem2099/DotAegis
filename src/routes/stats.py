"""Stats, cache, database, and performance endpoints."""
from fastapi import APIRouter, Depends
from starlette.responses import Response

from ..analyzer import analyzer, enhanced_cache
from ..database import db_manager
from ..security import get_api_key
from ..health import check_readiness, check_liveness, get_detailed_health

try:
    from ..performance_monitor import (
        get_performance_report, optimize_memory, detect_performance_bottlenecks
    )
except ImportError:
    def get_performance_report(): return {}
    def optimize_memory(): pass
    def detect_performance_bottlenecks(): return []

import os

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health")
def health():
    from datetime import datetime
    return {"status": "ok", "llm_ready": True,
            "timestamp": datetime.utcnow().isoformat(), "version": "2.1.3"}

@router.get("/health/detailed")
def detailed_health():
    return get_detailed_health()

@router.get("/readiness")
def readiness():
    return check_readiness()

@router.get("/liveness")
def liveness():
    return check_liveness()


# ── Metrics / Performance ─────────────────────────────────────────────────────
@router.get("/metrics")
def metrics():
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        fallback = (
            "# HELP llm_service_up LLM Service status\n"
            "# TYPE llm_service_up gauge\n"
            "llm_service_up 1\n"
        )
        return Response(content=fallback, media_type="text/plain; version=0.0.4; charset=utf-8")

@router.get("/performance/report")
def performance_report():
    return get_performance_report()

@router.post("/performance/optimize", dependencies=[Depends(get_api_key)])
def optimize_performance():
    optimize_memory()
    return {"status": "optimized", "message": "Memory optimization completed"}

@router.get("/performance/bottlenecks")
def detect_bottlenecks():
    issues = detect_performance_bottlenecks()
    return {
        "bottlenecks_detected": len(issues),
        "issues": issues,
        "recommendations": [r for r in [
            "Consider increasing memory limits"     if any("memory" in i.lower() for i in issues) else None,
            "Consider optimizing database queries"  if any("response time" in i.lower() for i in issues) else None,
            "Consider horizontal scaling"           if any("thread" in i.lower() for i in issues) else None,
        ] if r],
    }


# ── Main stats ────────────────────────────────────────────────────────────────
@router.get("/stats")
def get_stats():
    try:
        total = analyzer.cache_hits + analyzer.cache_misses
        hit_rate = analyzer.cache_hits / total if total else 0.0
        redis_stats = enhanced_cache.stats().get('l2', {})
        analytics_summary = db_manager.get_analytics_summary(days=7)
        return {
            "model":    analyzer.model.get_model_stats(),
            "training": analyzer.model.get_training_stats(),
            "cache": {
                "redis_status":       "connected" if redis_stats.get("available") else "disconnected",
                "cache_hit_rate":     hit_rate,
                "total_cache_requests": total,
                "cache_hits":         analyzer.cache_hits,
                "cache_misses":       analyzer.cache_misses,
                "redis_keys":         redis_stats.get("total_keys", 0),
                "redis_memory":       redis_stats.get("memory_used", "unknown"),
            },
            "database": {
                "analytics_summary":      analytics_summary,
                "training_samples_count": db_manager.count_training_samples(),
            },
            "performance": {
                "average_response_time": analytics_summary.get("average_processing_time_ms", 0),
            },
            "service": {
                "status":             "active",
                "version":            "2.1.3",
                "environment":        os.getenv('ENVIRONMENT', 'development'),
                "model_version":      analyzer.active_version,
                "ab_testing_enabled": analyzer.ab_testing_enabled,
            },
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Cache endpoints ───────────────────────────────────────────────────────────
@router.post("/cache/clear", dependencies=[Depends(get_api_key)])
def clear_cache():
    try:
        cleared_info = enhanced_cache.clear()
        return {"status": "cleared", "redis_keys_cleared": cleared_info.get('l2_cleared', 0),
                "message": "Cleared cache entries"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/cache/stats")
def get_cache_stats():
    try:
        total = analyzer.cache_hits + analyzer.cache_misses
        enhanced_stats = enhanced_cache.stats()
        return {
            "redis": enhanced_stats.get("l2", {}),
            "enhanced": enhanced_stats,
            "application": {
                "cache_hits":   analyzer.cache_hits,
                "cache_misses": analyzer.cache_misses,
                "hit_rate":     analyzer.cache_hits / total if total else 0.0,
                "total_requests": total,
            },
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Database endpoints ────────────────────────────────────────────────────────
@router.get("/analytics/summary")
def get_analytics_summary(days: int = 7):
    try:
        return db_manager.get_analytics_summary(days=days)
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/database/cleanup", dependencies=[Depends(get_api_key)])
def cleanup_database(days_to_keep: int = 90):
    try:
        ok = db_manager.cleanup_old_data(days_to_keep=days_to_keep)
        return {"status": "cleaned" if ok else "error",
                "message": f"Cleaned data older than {days_to_keep} days" if ok else "Cleanup failed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/database/training-samples", dependencies=[Depends(get_api_key)])
def get_training_samples(limit: int = 100, model_version: str = None):
    try:
        samples = db_manager.get_training_samples(limit=limit, model_version=model_version)
        return {"samples": samples, "count": len(samples),
                "limit": limit, "model_version": model_version}
    except Exception as e:
        return {"status": "error", "error": str(e)}
