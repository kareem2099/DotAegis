"""Request monitoring and security headers middleware."""
import time
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from .metrics import record_http_request
except ImportError:
    def record_http_request(*args, **kwargs): pass

from .logging_config import log_request, log_security_event

SKIP_LOG_PATHS = {"/health", "/metrics", "/readiness", "/liveness"}


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start     = time.time()
        method    = request.method
        path      = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        try:
            response = await call_next(request)
            elapsed  = time.time() - start
            record_http_request(method, path, response.status_code, elapsed)
            if path not in SKIP_LOG_PATHS:
                log_request(
                    request_id=getattr(request.state, 'request_id', 'unknown'),
                    method=method, path=path,
                    status_code=response.status_code,
                    duration_ms=int(elapsed * 1000),
                    client_ip=client_ip, user_agent=user_agent,
                )
            return response
        except Exception as e:
            elapsed = time.time() - start
            record_http_request(method, path, 500, elapsed)
            log_security_event("request_error", client_ip,
                               {"method": method, "path": path, "error": str(e)})
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    _HEADERS = {
        "X-Content-Type-Options":  "nosniff",
        "X-Frame-Options":         "DENY",
        "X-XSS-Protection":        "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy":         "strict-origin-when-cross-origin",
        "Permissions-Policy":      "geolocation=(), microphone=(), camera=()",
    }

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for header, value in self._HEADERS.items():
            response.headers[header] = value
        if "server" in response.headers:
            del response.headers["server"]  # hide server info
        return response
