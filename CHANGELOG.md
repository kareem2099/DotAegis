# Changelog

All notable changes to the DotAegis security service are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning Policy**: DotAegis version is synchronized with the DotEnvy VS Code Extension
> version it was designed to serve. Each major/minor bump here corresponds to the same release
> of DotEnvy that introduced the backend change.

---

## [2.1.3] - 2026-09-24

### **Security Hardening & Secret Rotation Architecture**

Synchronized with **DotEnvy v2.1.3**.

#### Security Hardening & Fixes
- **Dynamic Credential Rotation**: `register_or_get_client` now rotates and issues a fresh client secret upon re-registration instead of returning previously stored credentials, preventing credential harvesting.
- **Administrative Protection**: Added strict `X-API-KEY` / Bearer authorization to `POST /reset`, `POST /cache/clear`, `POST /database/cleanup`, and `/versions/*` endpoints.
- **Endpoint Deprecation**: Removed unauthenticated legacy `/auth/login` endpoint in favor of per-device HMAC-SHA256 signature verification.
- **Code Cleanliness**: Completely purged default fallback secrets from all test and training scripts.

---

## [2.1.2] - 2026-09-23

### **Dynamic Handshake & Zero Client Secret Architecture**

Synchronized with **DotEnvy v2.1.2**.

#### Authentication & Registration
- New `extension_clients` database table — tracks registered VS Code machines and their unique per-device client secrets.
- `POST /extension/register` — public handshake endpoint issuing unique credentials with strict IP rate limiting (max 10 registrations/hour per IP).
- In-memory credential caching with 5-minute TTL for high-throughput HMAC request verification.
- Dual-mode verification: prioritizes per-device credentials from DB/cache, falls back to legacy shared secret for older clients during transition.
- Individual client revocation support via `is_active` database flag.

---

## [2.1.0] - 2026-09-22

### **Aegis Backend** — Community Blacklist + Anti-Poisoning + Reputation System

Synchronized with **DotEnvy v2.1.0 (Aegis)**.

#### Community Blacklist System
- New `community_blacklist` table — stores promoted composite hashes synced to all clients
- New `staging_queue` table — holds candidate hashes pending consensus before promotion
- `POST /extension/blacklist/add` — clients submit hashes; server runs them through the staging pipeline
- `GET /extension/blacklist` — returns all promoted hashes for client-side cache sync (HMAC-authenticated GET)
- `POST /extension/blacklist/report_fp` — clients report false positives; 5 votes trigger automatic removal

#### Anti-Poisoning Protection
- Staging queue requires votes from ≥ 3 distinct machines before a hash is eligible for promotion
- Independent server-side LLM verification runs as a background task before any hash is promoted
- Malicious or low-quality submissions never enter the community blacklist directly

#### Reputation System
- New `machine_reputation` table — tracks correct/wrong submission history per VS Code machine ID
- Vote weight is dynamically calculated from historical accuracy: `> 0.9 → 2.0`, `> 0.7 → 1.0`, `> 0.5 → 0.5`, `≤ 0.5 → 0.0 (banned)`
- Machines that repeatedly submit false positives are progressively downweighted then banned
- False-positive reporters are penalized when FP consensus is reached

#### False Positive Voting
- New `fp_votes` table — one vote per machine per hash, prevents duplicate reports
- `increment_fp_votes()` — idempotent vote recording with automatic count return
- Automatic blacklist removal when FP vote count reaches threshold (5)

#### Client-Side Security Layers (L1–L4)
- **L1 — Local regex**: instant pattern matching for AWS, Stripe, GitHub, OpenAI, Google, Twilio, SendGrid keys — zero server cost
- **L2 — Community blacklist**: in-memory `Set<string>` synced at session start; composite hash lookup in O(1)
- **L3 — Shannon entropy gate**: values with entropy < 3.5 are classified `low` without any server call
- **L4 — LLM call**: only reached by ~20% of inputs after L1–L3 filtering
- `hashEntry(varName, value)` — composite hash using `SHA-256(varName:value[0:8]).slice(0,16)` for privacy-safe deduplication
- `syncHashToServer()` — fire-and-forget hash submission after any high-confidence detection
- `reportFalsePositive()` — removes hash from local in-memory blacklist on server confirmation
- `makeSignedGetRequest()` — dedicated HMAC-signed GET method for blacklist sync endpoint

#### Database
- `count_training_samples()` — efficient `COUNT(*)` query replacing full row fetch for stats endpoint
- `get_staging_entry()` — lightweight dict fetch for background task coordination
- `get_machine_reputation()`, `penalize_machine()` — reputation read/write methods
- `increment_fp_votes()` — idempotent FP vote recording
- `submit_hash()`, `mark_llm_verified()`, `_promote()` — full staging pipeline with session-safe JSON list mutation
- `get_blacklist_hashes()`, `remove_false_positive()` — blacklist read/write methods
- New indexes: `idx_blacklist_hash`, `idx_blacklist_promoted`, `idx_staging_hash`, `idx_staging_weight`

#### Performance & Caching
- Replaced `RedisCacheManager` (`redis_cache.py`) with unified `EnhancedCacheManager` — eliminates duplicate Redis connection pool
- L1 TTL cap raised from `TTL_SHORT` (5 min) to `TTL_MEDIUM` (1 hr) — reduces Redis round-trips for hot keys
- `pick_ttl()` now correctly propagates high-confidence TTL (24 hr) through both cache tiers

#### Auth & Rate Limiting
- `check_machine_rate_limit()` rewritten — cleans stale timestamps before appending, pops empty machine entries to prevent unbounded memory growth
- `verify_extension_signature()` — fixed `hmac.new()` to use positional arguments (Python compatibility)
- GET endpoint HMAC verification confirmed working with empty body (`timestamp.` prefix)

### Changed

- `stats.py` `/stats` endpoint now calls `db_manager.count_training_samples()` instead of fetching up to 1,000 rows and calling `len()`
- `health.py` `_check_cache()` migrated from `redis_cache` to `EnhancedCacheManager.stats()`
- `health.py` `_check_database()` uses `text("SELECT 1")` for SQLAlchemy 2.x compatibility
- `health.py` `DummyMetricsCollector` extended with `update_system_metrics()` stub
- `metrics.py` `get_metrics_dict()` and `_calculate_cache_hit_rate()` use `.get()` on Prometheus `_value` objects instead of raw attribute access
- `analyze.py` error response in `analyze_secret` restored — missing `return AnalyzeResponse(...)` syntax error fixed
- Service version string updated: `"1.0.0"` → `"2.1.0"` across all endpoints

### Security

- Community hash submissions are never trusted directly — consensus + independent LLM verification required
- Composite hash design (`varName:value[:8]`) prevents reverse-engineering of original secret values from stored hashes
- Replay attack window remains 5 minutes; constant-time HMAC comparison unchanged

### Removed

- `redis_cache.py` (`RedisCacheManager`) — superseded by `EnhancedCacheManager`; all imports updated

---

## [1.1.0] - 2026-04-11

### **Nexus Backend** — Service Modularization + DB Persistence + Feedback Loop

Synchronized with **DotEnvy v2.0.0 (Nexus)**.

#### Service Architecture
- Proper backpropagation through all transformer layers (replaced `_simple_gradient_update`)
- Adam optimizer with correct chain rule through attention + FFN blocks
- Model persistence: DB-primary (survives Railway redeploys) + local JSON fallback
- `/extension/feedback` endpoint — receives labeled samples from the extension, recalculates server-side features, stores to DB, triggers incremental training
- Expanded training dataset to 112+ labeled samples across high/medium/low/false_positive classes

---

## [1.0.0] - 2026-03-13

### **Initial Release** — Production LLM Backend

Synchronized with **DotEnvy v1.5.0**.

- Multi-stage Docker build with non-root `llmuser`
- Nginx reverse proxy with rate limiting and security headers
- FastAPI application with HMAC-authenticated extension endpoints (`extension_auth.py`)
- Custom transformer model with **35-feature extractor** (upgraded from 19)
- Two-tier cache: in-process LRU (500 items, 5min TTL) + Redis (1hr TTL) — 18.4x speedup measured
- PostgreSQL integration with SQLAlchemy + model weight persistence
- Prometheus metrics and structured JSON logging
- SSE streaming endpoint: `POST /extension/analyze/stream` — 5 progressive stages
- A/B testing and model versioning infrastructure (`/versions/*`, `/ab/*`)
- Circuit breaker on LLM analyzer with fallback to local feature scoring
- Health, readiness, and liveness probes
- HMAC-SHA256 authentication: replay attack protection (5min window), per-machine rate limiting (30 req/min)
- Service refactored from 600-line monolith into 9 focused modules

---

## Version History

| Version | Date       | Synchronized With     | Highlights |
|---------|------------|-----------------------|------------|
| **2.1.0** | 2026-09-22 | DotEnvy v2.1.0 Aegis  | Community blacklist, anti-poisoning, reputation system, L1–L4 layers |
| **1.1.0** | 2026-04-11 | DotEnvy v2.0.0 Nexus  | Real backprop, DB persistence, feedback loop, 112+ training samples |
| **1.0.0** | 2026-03-13 | DotEnvy v1.5.0        | Initial release: 35 features, HMAC auth, two-tier cache, SSE streaming |
