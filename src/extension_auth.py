"""
Extension Authentication Module
================================
Dynamic Per-Installation HMAC Authentication + Legacy Fallback.
No global secrets are exposed in the extension client code.

How it works:
1. Each installation registers via POST /extension/register to receive a unique client_secret.
2. Extension stores its unique secret securely in VS Code SecretStorage.
3. Extension signs every request body with HMAC-SHA256 using its device secret.
4. Server verifies the signature against the registered device credential in DB / cache.
5. Legacy fallback allows older 2.0.x / 2.1.0 extensions to function during transition.
6. Rate limiting is applied per Machine ID, plus strict IP rate-limiting on registration.
"""

import hmac
import hashlib
import time
import os
from typing import Optional, Tuple
from fastapi import HTTPException, Request, status
from collections import defaultdict

from .database import db_manager

# ─── Config ───────────────────────────────────────────────────────────────────
# Legacy fallback secret for older extension versions (if set)
EXTENSION_SHARED_SECRET = os.getenv("EXTENSION_SHARED_SECRET", "")
LEGACY_EXTENSION_SHARED_SECRET = os.getenv("LEGACY_EXTENSION_SHARED_SECRET", EXTENSION_SHARED_SECRET)

# Rate limiting configs
RATE_LIMIT_PER_MACHINE_PER_MINUTE = int(os.getenv("EXTENSION_RATE_LIMIT", "30"))
RATE_LIMIT_REGISTRATIONS_PER_IP_PER_HOUR = int(os.getenv("REGISTRATION_RATE_LIMIT_PER_IP", "10"))

# Simple in-memory rate limiters
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_ip_registration_store: dict[str, list[float]] = defaultdict(list)

# In-memory client credential cache: machine_id -> (client_secret, is_active, cached_at)
_client_credentials_cache: dict[str, Tuple[str, bool, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


# ─── Credential Cache Helpers ─────────────────────────────────────────────────

def get_cached_client_credentials(machine_id: str) -> Tuple[Optional[str], bool]:
    """Retrieve client secret and active status from memory cache or database."""
    now = time.time()
    if machine_id in _client_credentials_cache:
        secret, is_active, cached_at = _client_credentials_cache[machine_id]
        if now - cached_at < CACHE_TTL_SECONDS:
            return secret, is_active

    rec = db_manager.get_client_record(machine_id)
    if rec:
        secret = rec["client_secret"]
        is_active = rec["is_active"]
        _client_credentials_cache[machine_id] = (secret, is_active, now)
        return secret, is_active

    return None, True


def cache_registered_client(machine_id: str, secret: str) -> None:
    """Store newly registered credentials directly in cache."""
    _client_credentials_cache[machine_id] = (secret, True, time.time())


# ─── HMAC Verification ────────────────────────────────────────────────────────

def verify_extension_signature(
    body: bytes,
    signature: str,
    timestamp: str,
    secret: str,
) -> bool:
    """
    Verify that the request body matches the HMAC-SHA256 signature using secret.
    Signature = HMAC-SHA256(secret, timestamp + "." + body)
    """
    if not secret:
        return False

    # 1. Reject requests older than 5 minutes (replay attack protection)
    try:
        request_time = float(timestamp)
        if abs(time.time() - request_time) > 300:
            return False
    except (ValueError, TypeError):
        return False

    # 2. Compute expected signature (using safe positional arguments)
    message = f"{timestamp}.{body.decode('utf-8', errors='replace')}".encode()
    expected = hmac.new(
        secret.encode(),
        message,
        hashlib.sha256
    ).hexdigest()

    # 3. Constant-time comparison (prevents timing attacks)
    return hmac.compare_digest(expected, signature)


# ─── Rate Limiting ────────────────────────────────────────────────────────────

def check_machine_rate_limit(machine_id: str) -> bool:
    """
    Rate limit by Machine ID: max N requests per minute.
    Returns True if allowed, False if rate limited.
    """
    now = time.time()
    window_start = now - 60  # 1 minute window

    timestamps = [t for t in _rate_limit_store[machine_id] if t > window_start]
    if not timestamps:
        _rate_limit_store.pop(machine_id, None)
        _rate_limit_store[machine_id] = [now]
        return True

    if len(timestamps) >= RATE_LIMIT_PER_MACHINE_PER_MINUTE:
        _rate_limit_store[machine_id] = timestamps
        return False

    timestamps.append(now)
    _rate_limit_store[machine_id] = timestamps
    return True


def check_ip_registration_rate_limit(client_ip: str) -> bool:
    """
    Rate limit registrations by Client IP: max N requests per hour.
    Prevents mass registration / credential harvesting.
    """
    now = time.time()
    window_start = now - 3600  # 1 hour window

    timestamps = [t for t in _ip_registration_store[client_ip] if t > window_start]
    if len(timestamps) >= RATE_LIMIT_REGISTRATIONS_PER_IP_PER_HOUR:
        _ip_registration_store[client_ip] = timestamps
        return False

    timestamps.append(now)
    _ip_registration_store[client_ip] = timestamps
    return True


# ─── FastAPI Dependency ───────────────────────────────────────────────────────

async def verify_extension_request(request: Request) -> str:
    """
    FastAPI dependency that:
    1. Reads Machine-ID and HMAC signature headers
    2. Resolves client secret (per-device registration or legacy shared secret)
    3. Verifies HMAC signature
    4. Applies rate limiting per Machine ID
    
    Returns machine_id on success, raises HTTPException on failure.
    """
    timestamp = request.headers.get("X-Extension-Timestamp", "")
    signature = request.headers.get("X-Extension-Signature", "")
    machine_id = request.headers.get("X-Machine-ID", "")

    if not timestamp or not signature or not machine_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing extension authentication headers"
        )

    # Read body for signature verification
    body = await request.body()

    # 1. Resolve client credentials
    client_secret, is_active = get_cached_client_credentials(machine_id)

    if client_secret:
        if not is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Extension device credentials revoked"
            )
        if not verify_extension_signature(body, signature, timestamp, client_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid request signature for registered device"
            )
    else:
        # Fallback to legacy shared secret if configured (for older versions in grace period)
        legacy_secret = LEGACY_EXTENSION_SHARED_SECRET or EXTENSION_SHARED_SECRET
        if not legacy_secret or not verify_extension_signature(body, signature, timestamp, legacy_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unregistered device or invalid request signature"
            )

    # 2. Rate limit by machine ID
    if not check_machine_rate_limit(machine_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_PER_MACHINE_PER_MINUTE} requests/minute per machine"
        )

    return machine_id