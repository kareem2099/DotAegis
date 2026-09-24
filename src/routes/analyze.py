"""Analyze endpoints — /extension/analyze and /analyze."""
import hashlib
import secrets
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, status

from ..models import (
    AnalyzeRequest, AnalyzeResponse, FeedbackRequest,
    HashSubmitRequest, FPReportRequest,
    ExtensionRegisterRequest, ExtensionRegisterResponse
)
from ..analyzer import analyzer
from ..security import get_api_key, check_rate_limit
from ..extension_auth import (
    verify_extension_request,
    check_ip_registration_rate_limit,
    cache_registered_client
)
from ..database import db_manager

router = APIRouter()


@router.post("/extension/register", response_model=ExtensionRegisterResponse)
async def extension_register(
    body: ExtensionRegisterRequest,
    request: Request,
):
    """
    Registration endpoint for VS Code extension installations.
    Issues a unique per-device client_secret for HMAC signing.
    Protected by strict IP rate limiting (max 10/hr per IP).
    """
    client_ip = request.client.host if request.client else "unknown"

    if not check_ip_registration_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Registration rate limit exceeded (max 10 registrations per hour per IP)"
        )

    try:
        client_secret, is_new, created_at = db_manager.register_or_get_client(
            machine_id=body.machine_id,
            vscode_version=body.vscode_version or "",
            extension_version=body.extension_version or "",
            client_ip=client_ip
        )
        cache_registered_client(body.machine_id, client_secret)

        return ExtensionRegisterResponse(
            status="ok",
            machine_id=body.machine_id,
            client_secret=client_secret,
            created_at=created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


def _build_response(confidence_level: str, secret_value: str, context: str,
                    request_id: str, extra_reasoning: list = None) -> AnalyzeResponse:
    entropy  = analyzer._calculate_entropy(secret_value)
    category = analyzer._categorize_secret(secret_value)
    risk     = analyzer._assess_risk_level(secret_value, context)
    is_likely = (confidence_level != 'low') or (entropy > 3.5)
    reasoning = [f"Enhanced confidence: {confidence_level}", f"Entropy: {entropy:.2f}"]
    if extra_reasoning:
        reasoning.extend(extra_reasoning)
    return AnalyzeResponse(
        enhanced_confidence=confidence_level,
        method="llm_hybrid",
        is_likely_secret=is_likely,
        category=category,
        risk_level=risk,
        reasoning=reasoning,
        request_id=request_id,
    )


@router.post("/extension/analyze", response_model=AnalyzeResponse)
async def extension_analyze(
    request: AnalyzeRequest,
    request_obj: Request,
    machine_id: str = Depends(verify_extension_request),
):
    """HMAC-authenticated endpoint for the VS Code extension."""
    request_id = secrets.token_hex(8)
    try:
        confidence_level = analyzer.calculate_enhanced_confidence(
            request.secret_value, request.context, "low", request.variable_name
        )
        return _build_response(confidence_level, request.secret_value,
                               request.context, request_id)
    except Exception as e:
        return AnalyzeResponse(
            enhanced_confidence="low", method="error", is_likely_secret=False,
            category="Unknown", risk_level="low",
            reasoning=[f"Error: {str(e)}"], request_id=request_id,
        )


@router.post("/extension/feedback")
async def extension_feedback(
    request: FeedbackRequest,
    machine_id: str = Depends(verify_extension_request),
):
    """HMAC-authenticated endpoint for sending feedback samples from the extension."""
    stored_count = 0
    for sample in request.samples:
        try:
            # Server-side feature recalculation for consistency
            features_vector = analyzer.extract_features(
                sample.secret_value, sample.context, sample.variable_name
            )
            features_list = features_vector.tolist()

            db_manager.store_training_sample(
                secret_hash=hashlib.sha256(sample.secret_value.encode()).hexdigest(),
                context_hash=hashlib.sha256(sample.context.encode()).hexdigest(),
                features=features_list,
                label=sample.label,
                user_action=sample.user_action,
                confidence=0.5,
                model_version=analyzer.active_version,
            )

            # Also add to the in-memory model training pool
            analyzer.model.add_training_sample(sample.secret_value, features_list, sample.label)
            stored_count += 1
        except Exception as e:
            print(f"Failed to process feedback sample from machine {machine_id}: {e}")

    # Trigger training if we have enough new samples (consistent with train.py)
    if stored_count > 0 and len(analyzer.model.training_samples) >= 5:
        analyzer.model.train(analyzer.model.training_samples[-10:], epochs=1)
        analyzer.save_model()

    return {"status": "ok", "stored_samples": stored_count}


@router.post("/analyze", response_model=AnalyzeResponse,
             dependencies=[Depends(get_api_key), Depends(check_rate_limit)])
def analyze_secret(request: AnalyzeRequest, request_obj: Request):
    """API-key authenticated endpoint for internal/admin use."""
    start_time = datetime.utcnow()
    request_id = secrets.token_hex(8)
    client_ip  = request_obj.client.host if request_obj.client else "unknown"
    user_agent = request_obj.headers.get("user-agent", "")

    try:
        confidence_level = analyzer.calculate_enhanced_confidence(
            request.secret_value, request.context, "low", request.variable_name
        )
        total = analyzer.cache_hits + analyzer.cache_misses
        extra = [f"Cache hit rate: {analyzer.cache_hits / total:.1%}"] if total > 0 else []
        response = _build_response(confidence_level, request.secret_value,
                                   request.context, request_id, extra)

        # Analytics (non-blocking)
        try:
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            db_manager.store_analysis_request(
                request_id=request_id, client_ip=client_ip, user_agent=user_agent,
                secret_type=response.category, risk_level=response.risk_level,
                confidence=float(confidence_level == 'high'),
                processing_time_ms=processing_time_ms,
                cache_hit=(total > 0 and analyzer.cache_hits > 0),
                api_key_hash=hashlib.sha256(
                    str(getattr(request_obj.state, 'api_key', '')).encode()
                ).hexdigest(),
            )
        except Exception as db_err:
            print(f"Analytics storage failed: {db_err}")

        return response

    except Exception as e:
        try:
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            db_manager.store_analysis_request(
                request_id=request_id, client_ip=client_ip, user_agent=user_agent,
                secret_type="error", risk_level="unknown", confidence=0.0,
                processing_time_ms=processing_time_ms, cache_hit=False, api_key_hash="",
            )
        except Exception:
            pass

        return AnalyzeResponse(
            enhanced_confidence="low", method="error", is_likely_secret=False,
            category="Unknown", risk_level="low",
            reasoning=[f"Server Error: {str(e)}"], request_id=request_id,
        )


# ─── Community Blacklist ───────────────────────────────────────────────────────

@router.post("/extension/blacklist/add")
async def blacklist_add(
    body: HashSubmitRequest,
    machine_id: str = Depends(verify_extension_request),
):
    """Receive a composite hash from a client and run it through staging."""
    weight = _get_machine_weight(machine_id)
    if weight == 0.0:
        return {"status": "banned"}

    status = db_manager.submit_hash(body.hash, machine_id, weight)

    # If it hit the vote threshold, verify independently with our own LLM
    if status == "staged":
        staging = db_manager.get_staging_entry(body.hash)
        if staging and staging["vote_weight"] >= 3.0 and not staging["llm_verified"]:
            asyncio.create_task(_verify_and_promote(body.hash))

    return {"status": status}


@router.get("/extension/blacklist")
async def blacklist_sync(
    machine_id: str = Depends(verify_extension_request),
):
    """Return all promoted hashes for client-side cache."""
    hashes = db_manager.get_blacklist_hashes()
    return {"hashes": hashes, "count": len(hashes)}


@router.post("/extension/blacklist/report_fp")
async def blacklist_report_fp(
    body: FPReportRequest,
    machine_id: str = Depends(verify_extension_request),
):
    """Client reports a false positive. 5 votes → remove from blacklist."""
    count = db_manager.increment_fp_votes(body.hash, machine_id)
    if count >= 5:
        db_manager.remove_false_positive(body.hash)
        db_manager.penalize_machine(machine_id, correct=False)
        return {"status": "removed", "fp_votes": count}
    return {"status": "recorded", "fp_votes": count}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _get_machine_weight(machine_id: str) -> float:
    rep = db_manager.get_machine_reputation(machine_id)
    total = rep.get("correct", 0) + rep.get("wrong", 0)
    if total < 5:
        return 1.0
    accuracy = rep["correct"] / total if total > 0 else 1.0
    if accuracy > 0.9: return 2.0
    if accuracy > 0.7: return 1.0
    if accuracy > 0.5: return 0.5
    return 0.0


async def _verify_and_promote(hash_val: str):
    """Background task: server verifies hash with its own LLM before promoting."""
    try:
        # Note: We use a placeholder here for the evaluation logic. 
        # In a real scenario, we'd need context to verify a hash properly.
        dummy_value = "x" * 32  # placeholder high-entropy shape
        confidence = analyzer.calculate_enhanced_confidence(
            dummy_value, f"hash_verification:{hash_val}", "low"
        )
        if confidence in ("high", "critical", "medium"):
            db_manager.mark_llm_verified(hash_val)
    except Exception as e:
        print(f"Background verification failed for {hash_val}: {e}")
