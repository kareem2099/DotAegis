#!/usr/bin/env python3
"""
python-llm Local Test Suite
============================
Run before git push to Railway.

Usage:
    python test_local.py                    # tests against localhost:8080
    python test_local.py --url http://localhost:8000
    python test_local.py --url https://your-railway-url.up.railway.app
"""

import sys
import json
import hmac
import hashlib
import time
import argparse
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("❌ requests not installed — run: pip install requests")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the service")
parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key for /analyze")
parser.add_argument("--secret", default=os.getenv("EXTENSION_SHARED_SECRET", ""), help="HMAC shared secret")
args = parser.parse_args()

BASE    = args.url.rstrip("/")
API_KEY = args.api_key
SECRET  = args.secret

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def run(label, fn):
    try:
        ok, msg = fn()
        status = PASS if ok else FAIL
        print(f"  {status} {label}{(' — ' + msg) if msg else ''}")
        results.append((ok, label))
    except Exception as e:
        print(f"  {FAIL} {label} — exception: {e}")
        results.append((False, label))


# ── HMAC helper (mirrors llmAnalyzer.ts) ─────────────────────────────────────
def make_hmac_headers(body: dict, machine_id: str = "test-machine-001"):
    """Returns (headers, body_bytes) — must send the SAME bytes we signed."""
    body_bytes = json.dumps(body, separators=(',', ':'), sort_keys=True).encode()
    timestamp  = str(time.time())
    message    = f"{timestamp}.".encode() + body_bytes
    signature  = hmac.new(SECRET.encode(), message, hashlib.sha256).hexdigest()
    headers = {
        "X-Extension-Timestamp": timestamp,
        "X-Extension-Signature": signature,
        "X-Machine-ID":          machine_id,
        "Content-Type":          "application/json",
    }
    return headers, body_bytes


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  python-llm Test Suite")
print(f"  Target: {BASE}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")


# ── 1. Health ─────────────────────────────────────────────────────────────────
print("1. Health Checks")

def test_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    d = r.json()
    ok = r.status_code == 200 and d.get("status") == "ok" and d.get("llm_ready") is True
    return ok, f"status={d.get('status')}, llm_ready={d.get('llm_ready')}"
run("GET /health", test_health)

def test_readiness():
    r = requests.get(f"{BASE}/readiness", timeout=10)
    return r.status_code == 200, f"HTTP {r.status_code}"
run("GET /readiness", test_readiness)

def test_liveness():
    r = requests.get(f"{BASE}/liveness", timeout=10)
    return r.status_code == 200, f"HTTP {r.status_code}"
run("GET /liveness", test_liveness)


# ── 2. /extension/analyze (HMAC auth) ────────────────────────────────────────
print("\n2. Extension Analyze — HMAC Auth")

PAYLOADS = [
    {
        "label": "Stripe API key (expect: high)",
        "body":  {"secret_value": "sk-live-abc123XYZdef456GHIjkl789MNO", "context": "const stripeKey = \"sk-live-abc123XYZdef456GHIjkl789MNO\";", "variable_name": "stripeKey"},
        "expect_high": True,
    },
    {
        "label": "GitHub token (expect: high)",
        "body":  {"secret_value": "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "context": "GITHUB_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "variable_name": "GITHUB_TOKEN"},
        "expect_high": True,
    },
    {
        "label": "Plain string (expect: low/medium)",
        "body":  {"secret_value": "hello-world", "context": "const greeting = \"hello-world\";", "variable_name": "greeting"},
        "expect_high": False,
    },
    {
        "label": "AWS key pattern (expect: high)",
        "body":  {"secret_value": "AKIAIOSFODNN7EXAMPLE", "context": "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "variable_name": "AWS_ACCESS_KEY_ID"},
        "expect_high": True,
    },
]

def test_extension_analyze(payload):
    headers, body_bytes = make_hmac_headers(payload["body"])
    r = requests.post(f"{BASE}/extension/analyze", data=body_bytes, headers=headers, timeout=15)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    d = r.json()
    confidence = d.get("enhanced_confidence", "")
    is_high    = (confidence == "high")
    category   = d.get("category", "?")
    expected   = "high" if payload["expect_high"] else "low/medium"
    match      = (is_high == payload["expect_high"])
    return match, f"confidence={confidence}, category={category} (expected ~{expected})"

for p in PAYLOADS:
    run(p["label"], lambda p=p: test_extension_analyze(p))

def test_extension_no_auth():
    r = requests.post(f"{BASE}/extension/analyze",
                      json={"secret_value": "test", "context": "test"},
                      timeout=10)
    return r.status_code in (401, 403, 422), f"HTTP {r.status_code} (want 401/403)"
run("Rejects unauthenticated request", test_extension_no_auth)

def test_extension_bad_signature():
    body = {"secret_value": "test", "context": "test"}
    headers, body_bytes = make_hmac_headers(body)
    headers["X-Extension-Signature"] = "badsignature"
    r = requests.post(f"{BASE}/extension/analyze", data=body_bytes, headers=headers, timeout=10)
    return r.status_code in (401, 403), f"HTTP {r.status_code} (want 401/403)"
run("Rejects bad HMAC signature", test_extension_bad_signature)


# ── 3. /analyze (API key auth) ────────────────────────────────────────────────
print("\n3. /analyze — API Key Auth")

if not API_KEY:
    print(f"  {WARN} Skipping — no API_KEY set (use --api-key or $API_KEY)")
else:
    def test_analyze_auth():
        payload = {"secret_value": "sk-live-testkey123", "context": "const key = 'sk-live-testkey123';"}
        r = requests.post(f"{BASE}/analyze", json=payload,
                          headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
        return r.status_code == 200, f"HTTP {r.status_code}"
    run("POST /analyze with valid API key", test_analyze_auth)

    def test_analyze_no_auth():
        r = requests.post(f"{BASE}/analyze",
                          json={"secret_value": "test", "context": "test"}, timeout=10)
        return r.status_code == 401, f"HTTP {r.status_code} (want 401)"
    run("POST /analyze rejects no auth", test_analyze_no_auth)


# ── 4. Feature count validation (35 features) ────────────────────────────────
print("\n4. Feature Extractor — 35 Features")

def test_feature_count():
    """Import feature_extractor directly and check output shape."""
    try:
        import importlib.util, pathlib
        # Try to find it relative to CWD (run from python-llm root)
        candidates = [
            "src/feature_extractor.py",
            "feature_extractor.py",
        ]
        spec = None
        for c in candidates:
            p = pathlib.Path(c)
            if p.exists():
                spec = importlib.util.spec_from_file_location("feature_extractor", p)
                break
        if not spec:
            return None, "feature_extractor.py not found locally (skip)"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arr = mod.extract("sk-live-abc123", "const key = 'sk-live-abc123'", "apiKey")
        n   = len(arr)
        return n == 35, f"got {n} features (want 35)"
    except Exception as e:
        return False, str(e)

result, msg = test_feature_count()
if result is None:
    print(f"  {WARN} Feature count check skipped — {msg}")
else:
    status = PASS if result else FAIL
    print(f"  {status} feature_extractor returns 35 features — {msg}")
    results.append((result if result is not None else True, "35 features"))


# ── 5. Cache ──────────────────────────────────────────────────────────────────
print("\n5. Cache")

def test_cache_stats():
    r = requests.get(f"{BASE}/cache/stats", timeout=10)
    return r.status_code == 200, f"HTTP {r.status_code}"
run("GET /cache/stats", test_cache_stats)

def test_cache_speedup():
    """Second identical request should be faster (cache hit)."""
    body = {"secret_value": "ghp_speedtest123456789abcdef", "context": "GITHUB_TOKEN=ghp_speedtest123456789abcdef", "variable_name": "GITHUB_TOKEN"}

    headers1, body_bytes = make_hmac_headers(body)
    t0 = time.perf_counter()
    requests.post(f"{BASE}/extension/analyze", data=body_bytes, headers=headers1, timeout=15)
    t1 = time.perf_counter()

    headers2, body_bytes2 = make_hmac_headers(body)
    t2 = time.perf_counter()
    requests.post(f"{BASE}/extension/analyze", data=body_bytes2, headers=headers2, timeout=15)
    t3 = time.perf_counter()

    first_ms  = (t1 - t0) * 1000
    second_ms = (t3 - t2) * 1000
    speedup   = first_ms / max(second_ms, 0.1)
    return speedup >= 1.2, f"1st={first_ms:.0f}ms, 2nd={second_ms:.0f}ms, speedup={speedup:.1f}x"
run("Cache speedup on repeated request", test_cache_speedup)


# ── 6. Stats ──────────────────────────────────────────────────────────────────
print("\n6. Stats Endpoint")

def test_stats():
    r = requests.get(f"{BASE}/stats", timeout=10)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    d = r.json()
    has_model   = "model" in d
    has_cache   = "cache" in d
    has_service = "service" in d
    num_features = d.get("model", {}).get("num_features", 0)
    return has_model and has_cache and has_service, \
           f"model={has_model}, cache={has_cache}, num_features={num_features}"
run("GET /stats", test_stats)


# ── 7. Streaming ──────────────────────────────────────────────────────────────
print("\n7. Streaming (SSE)")

def test_streaming():
    body = {"secret_value": "sk-live-streamtest123", "context": "const key = 'sk-live-streamtest123'"}
    headers, body_bytes = make_hmac_headers(body)
    try:
        r = requests.post(f"{BASE}/extension/analyze/stream", data=body_bytes,
                          headers=headers, stream=True, timeout=20)
        if r.status_code == 404:
            return False, "404 — streaming route not registered"
        events = []
        for line in r.iter_lines(chunk_size=None):
            if line and line.startswith(b"data:"):
                events.append(json.loads(line[5:].strip()))
            if len(events) >= 3:
                break
        stages = [e.get("stage") for e in events]
        return len(events) >= 2, f"received {len(events)} SSE events, stages={stages}"
    except Exception as e:
        return False, str(e)
run("POST /extension/analyze/stream", test_streaming)


# ── 8. Input validation ───────────────────────────────────────────────────────
print("\n8. Input Validation")

def test_too_long_input():
    body = {"secret_value": "x" * 20000, "context": "test"}
    headers, body_bytes = make_hmac_headers(body)
    r = requests.post(f"{BASE}/extension/analyze", data=body_bytes, headers=headers, timeout=10)
    return r.status_code == 422, f"HTTP {r.status_code} (want 422)"
run("Rejects secret_value > 10000 chars", test_too_long_input)

def test_empty_input():
    body = {"secret_value": "", "context": "test"}
    headers, body_bytes = make_hmac_headers(body)
    r = requests.post(f"{BASE}/extension/analyze", data=body_bytes, headers=headers, timeout=10)
    return r.status_code == 422, f"HTTP {r.status_code} (want 422)"
run("Rejects empty secret_value", test_empty_input)


# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
    print("\n  Failed tests:")
    for ok, label in results:
        if not ok:
            print(f"    {FAIL} {label}")
else:
    print(" 🎉")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)