"""
Streaming Analysis Endpoint
============================

Adds  POST /extension/analyze/stream  that returns Server-Sent Events (SSE).

The client receives incremental progress so the UI can show live feedback
instead of waiting for the full analysis to complete.

Stream format (each line is a JSON event):
  data: {"stage": "pattern_match",  "result": {...}, "progress": 25}
  data: {"stage": "entropy",        "result": {...}, "progress": 50}
  data: {"stage": "context",        "result": {...}, "progress": 75}
  data: {"stage": "final",          "result": {...}, "progress": 100}

Usage (add to service.py):
  from .streaming import add_streaming_routes
  add_streaming_routes(app, analyzer, verify_extension_request)
"""

import json
import asyncio
import secrets as secrets_mod
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class StreamAnalyzeRequest(BaseModel):
    secret_value: str
    context: str
    variable_name: Optional[str] = None


def _sse(data: dict) -> str:
    """Format a single SSE event."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_analysis(
    secret_value: str,
    context: str,
    variable_name: Optional[str],
    analyzer,          # LLMAnalyzer instance
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE events as analysis stages complete.
    Each stage adds insight so the client can update the UI progressively.
    """
    from .feature_extractor import extract, SECRET_PATTERNS
    import re
    import math

    # ── Stage 1: Pattern matching (fast, ~1ms) ────────────────────────────────
    await asyncio.sleep(0)   # yield control so FastAPI can flush

    pattern_matches = []
    best_score = 0.0
    for pattern, score, name in SECRET_PATTERNS:
        if re.search(pattern, secret_value, re.IGNORECASE | re.MULTILINE):
            pattern_matches.append({'name': name, 'score': score})
            if score > best_score:
                best_score = score

    yield _sse({
        'stage':    'pattern_match',
        'progress': 20,
        'result': {
            'matched_patterns': pattern_matches,
            'best_score':       best_score,
            'early_verdict':    'likely_secret' if best_score >= 0.9 else 'uncertain',
        },
        'request_id': request_id,
    })

    # ── Stage 2: Entropy analysis ─────────────────────────────────────────────
    await asyncio.sleep(0)

    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        return -sum((v / n) * math.log2(v / n) for v in freq.values())

    entropy = _entropy(secret_value)
    unique_ratio = len(set(secret_value)) / max(1, len(secret_value))
    length = len(secret_value)

    yield _sse({
        'stage':    'entropy',
        'progress': 40,
        'result': {
            'entropy':      round(entropy, 3),
            'unique_ratio': round(unique_ratio, 3),
            'length':       length,
            'entropy_verdict': (
                'high_entropy' if entropy > 4.0 else
                'medium_entropy' if entropy > 2.5 else
                'low_entropy'
            ),
        },
        'request_id': request_id,
    })

    # ── Stage 3: Context analysis ─────────────────────────────────────────────
    await asyncio.sleep(0)

    ctx_lower = context.lower()
    HIGH_RISK = ['password', 'secret', 'private_key', 'api_key', 'auth_token']
    MED_RISK  = ['token', 'key', 'auth', 'api', 'bearer', 'access']

    high_hits = [kw for kw in HIGH_RISK if kw in ctx_lower]
    med_hits  = [kw for kw in MED_RISK if kw in ctx_lower]
    context_risk_score = min(1.0, len(high_hits) * 0.4 + len(med_hits) * 0.2)

    yield _sse({
        'stage':    'context',
        'progress': 65,
        'result': {
            'high_risk_keywords': high_hits,
            'medium_risk_keywords': med_hits,
            'context_risk_score': round(context_risk_score, 3),
            'has_assignment': '=' in context,
            'has_quotes': any(q in context for q in ('"', "'", '`')),
        },
        'request_id': request_id,
    })

    # ── Stage 4: Full LLM analysis ────────────────────────────────────────────
    await asyncio.sleep(0)

    try:
        confidence_level = analyzer.calculate_enhanced_confidence(
            secret_value, context, 'low', variable_name
        )
    except Exception as e:
        confidence_level = 'low'

    category  = analyzer._categorize_secret(secret_value)
    risk      = analyzer._assess_risk_level(secret_value, context)
    is_likely = (confidence_level != 'low') or (entropy > 3.5) or (best_score >= 0.9)

    yield _sse({
        'stage':    'llm_analysis',
        'progress': 85,
        'result': {
            'confidence':  confidence_level,
            'category':    category,
            'risk_level':  risk,
        },
        'request_id': request_id,
    })

    # ── Stage 5: Final verdict ────────────────────────────────────────────────
    await asyncio.sleep(0)

    reasoning = []
    if pattern_matches:
        reasoning.append(f"Matched known pattern: {pattern_matches[0]['name']}")
    if entropy > 4.0:
        reasoning.append(f"High entropy ({entropy:.2f}) indicates random/generated value")
    if high_hits:
        reasoning.append(f"Context contains high-risk keywords: {', '.join(high_hits)}")
    if length >= 20:
        reasoning.append(f"Length ({length}) consistent with API keys/tokens")
    if not reasoning:
        reasoning.append("No strong secret indicators found")

    yield _sse({
        'stage':    'final',
        'progress': 100,
        'result': {
            'enhanced_confidence': confidence_level,
            'method':             'llm_hybrid_stream',
            'is_likely_secret':   is_likely,
            'category':           category,
            'risk_level':         risk,
            'reasoning':          reasoning,
        },
        'request_id': request_id,
        'timestamp': datetime.utcnow().isoformat(),
    })


def add_streaming_routes(app, analyzer, verify_extension_request):
    """
    Call this from service.py after creating the FastAPI `app`:

        from .streaming import add_streaming_routes
        add_streaming_routes(app, analyzer, verify_extension_request)
    """

    @app.post("/extension/analyze/stream")
    async def extension_analyze_stream(
        request: StreamAnalyzeRequest,
        request_obj: Request,
        machine_id: str = Depends(verify_extension_request),
    ):
        """
        Streaming analysis endpoint — returns SSE events.

        TypeScript client example:
            const es = await fetch('/extension/analyze/stream', {
                method: 'POST',
                headers: { ...hmacHeaders, 'Content-Type': 'application/json' },
                body: JSON.stringify({ secret_value, context }),
            });
            const reader = es.body.getReader();
            // Read chunks and parse `data: {...}` lines
        """
        request_id = secrets_mod.token_hex(8)

        return StreamingResponse(
            _stream_analysis(
                request.secret_value,
                request.context,
                request.variable_name,
                analyzer,
                request_id,
            ),
            media_type='text/event-stream',
            headers={
                'Cache-Control':     'no-cache',
                'X-Accel-Buffering': 'no',   # disable Nginx buffering
                'Connection':        'keep-alive',
            },
        )