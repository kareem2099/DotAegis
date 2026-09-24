"""
Enhanced Cache Manager
======================

Two-tier caching:
  L1 — In-process LRU  (fast, no network, ~500 items, 5 min TTL)
  L2 — Redis           (shared across workers, 1 hour TTL)

If Redis is down the service degrades gracefully to L1-only.
"""

import time
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Optional
import os

# ─── TTL tiers (seconds) ──────────────────────────────────────────────────────
TTL_SHORT  = 5 * 60        # 5 min  — volatile patterns that may be retested
TTL_MEDIUM = 60 * 60       # 1 hour — default analysis results
TTL_LONG   = 24 * 60 * 60  # 24 hrs — well-known library patterns (high confidence)


class LRUCache:
    """Thread-safe in-process LRU cache with per-item TTL."""

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            value, expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                self.misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: int = TTL_MEDIUM):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time() + ttl)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # evict oldest

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / max(1, total),
            }


class EnhancedCacheManager:
    """
    Two-tier cache: L1 (LRU in-process) → L2 (Redis).

    Usage:
        cache = EnhancedCacheManager()
        key   = cache.make_key(secret, context, variable_name)
        value = cache.get(key)
        if value is None:
            value = expensive_analysis(...)
            cache.set(key, value, ttl=cache.pick_ttl(value))
    """

    def __init__(self, redis_url: Optional[str] = None, l1_max_size: int = 500):
        self._l1 = LRUCache(max_size=l1_max_size)
        self._redis = None
        self._redis_available = False
        self._connect_redis(redis_url or os.getenv('REDIS_URL', ''))

    # ── Redis connection ───────────────────────────────────────────────────────

    def _connect_redis(self, url: str):
        if not url:
            print("⚠️  No REDIS_URL — running L1 cache only")
            return
        try:
            import redis
            client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            self._redis = client
            self._redis_available = True
            print("✅ Redis cache connected")
        except Exception as e:
            print(f"⚠️  Redis unavailable ({e}) — L1 cache only")

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(secret: str, context: str, variable_name: Optional[str] = None) -> str:
        """Privacy-safe cache key (never stores raw secrets)."""
        raw = f"{secret}|{context}|{variable_name or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        # L1 first
        value = self._l1.get(key)
        if value is not None:
            return value

        # L2 (Redis)
        if self._redis_available:
            try:
                raw = self._redis.get(key)
                if raw:
                    value = json.loads(raw)
                    # Backfill L1
                    self._l1.set(key, value, ttl=TTL_SHORT)
                    return value
            except Exception:
                self._redis_available = False  # fail-open: degrade silently

        return None

    def set(self, key: str, value: Any, ttl: int = TTL_MEDIUM):
        self._l1.set(key, value, ttl=min(ttl, TTL_MEDIUM))

        if self._redis_available:
            try:
                self._redis.setex(key, ttl, json.dumps(value))
            except Exception:
                self._redis_available = False

    def delete(self, key: str):
        self._l1.delete(key)
        if self._redis_available:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def clear(self) -> dict:
        l1_cleared = self._l1.clear()
        l2_cleared = 0
        if self._redis_available:
            try:
                keys = self._redis.keys('*')
                if keys:
                    l2_cleared = self._redis.delete(*keys)
            except Exception:
                pass
        return {'l1_cleared': l1_cleared, 'l2_cleared': l2_cleared}

    @staticmethod
    def pick_ttl(confidence_level: str) -> int:
        """High-confidence results are cached longer."""
        return {
            'high':           TTL_LONG,
            'medium':         TTL_MEDIUM,
            'low':            TTL_SHORT,
            'false_positive': TTL_LONG,
        }.get(confidence_level, TTL_MEDIUM)

    def stats(self) -> dict:
        l1 = self._l1.stats()
        l2: dict = {'available': self._redis_available}
        if self._redis_available:
            try:
                info = self._redis.info('memory')
                l2['memory_used'] = info.get('used_memory_human', 'unknown')
                l2['total_keys'] = self._redis.dbsize()
            except Exception:
                pass
        return {'l1': l1, 'l2': l2}