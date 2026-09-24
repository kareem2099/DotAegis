"""Security manager, API key auth, JWT, and rate limiting."""
import os
import hashlib
import hmac
from typing import Optional
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class SecurityManager:
    def __init__(self):
        self.api_keys: set = set()
        self.jwt_secret = os.getenv('JWT_SECRET', 'default-jwt-secret-change-in-production')
        self.rate_limits: dict = {}
        self.max_requests_per_minute = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '60'))

    def load_api_keys(self):
        api_key_env = os.getenv('API_KEY', '')
        if api_key_env:
            self.api_keys = set(k.strip() for k in api_key_env.split(',') if k.strip())

    def verify_api_key(self, api_key: str) -> bool:
        return api_key in self.api_keys

    def generate_jwt_token(self, username: str, expires_delta: timedelta = None) -> str:
        if expires_delta is None:
            expires_delta = timedelta(hours=1)
        expire = datetime.utcnow() + expires_delta
        payload = {"sub": username, "exp": expire, "iat": datetime.utcnow(), "type": "access"}
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_jwt_token(self, token: str) -> Optional[dict]:
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def check_rate_limit(self, client_ip: str) -> bool:
        current_time = datetime.utcnow()
        minute_key   = current_time.strftime("%Y-%m-%d %H:%M")

        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {}

        client_limits = self.rate_limits[client_ip]
        cutoff = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)

        for ts in list(client_limits):
            if datetime.fromisoformat(ts) < cutoff:
                del client_limits[ts]
                
        if not client_limits:
            del self.rate_limits[client_ip]
            self.rate_limits[client_ip] = {}
            client_limits = self.rate_limits[client_ip]

        client_limits.setdefault(minute_key, 0)
        if client_limits[minute_key] >= self.max_requests_per_minute:
            return False
        client_limits[minute_key] += 1
        return True

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        return hmac.compare_digest(self.hash_password(password), hashed)


# ── Singleton ──────────────────────────────────────────────────────────────────
security_manager = SecurityManager()
security_scheme  = HTTPBearer(auto_error=False)


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
    if credentials.scheme.lower() == "bearer":
        payload = security_manager.verify_jwt_token(credentials.credentials)
        if payload:
            return payload
        if security_manager.verify_api_key(credentials.credentials):
            return credentials.credentials
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication credentials")


async def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded. Please try again later.")
    return True
