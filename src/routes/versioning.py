"""Model versioning and A/B testing endpoints."""
from fastapi import APIRouter, Depends
from ..analyzer import analyzer
from ..security import get_api_key

router = APIRouter()


# ── Versions ──────────────────────────────────────────────────────────────────
@router.post("/versions/create", dependencies=[Depends(get_api_key)])
def create_model_version(version_name: str):
    try:
        ok = analyzer.create_model_version(version_name)
        return {"status": "created", "version": version_name} if ok \
            else {"status": "error", "message": f"Version '{version_name}' already exists"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/versions/switch", dependencies=[Depends(get_api_key)])
def switch_model_version(version_name: str):
    try:
        ok = analyzer.switch_model_version(version_name)
        return {"status": "switched", "active_version": version_name} if ok \
            else {"status": "error", "message": f"Version '{version_name}' not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/versions")
def get_model_versions():
    try:
        return analyzer.get_model_version_info()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── A/B testing ───────────────────────────────────────────────────────────────
@router.post("/ab/enable", dependencies=[Depends(get_api_key)])
def enable_ab_testing(version_a: str = "default", version_b: str = "default"):
    try:
        ok = analyzer.enable_ab_testing(version_a, version_b)
        return {"status": "enabled", "group_a": version_a, "group_b": version_b} if ok \
            else {"status": "error", "message": "Invalid version names"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/ab/disable", dependencies=[Depends(get_api_key)])
def disable_ab_testing():
    try:
        analyzer.disable_ab_testing()
        return {"status": "disabled", "active_version": "default"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/ab/results")
def get_ab_test_results():
    try:
        return analyzer.get_ab_test_results()
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/ab/record", dependencies=[Depends(get_api_key)])
def record_prediction_result(version: str, correct: bool):
    try:
        analyzer.record_prediction_result(version, correct)
        return {"status": "recorded", "version": version, "correct": correct}
    except Exception as e:
        return {"status": "error", "error": str(e)}
