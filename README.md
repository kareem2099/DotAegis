# DotAegis

DotSuite's FastAPI service for classifying potential credentials. Used by the
[DotEnvy VS Code extension](https://github.com/kareem2099/dotenvy).

**Version 2.2.4 · feature schema 2 · Python 3.11+**

## Detection and learning

DotEnvy **2.2.4+** runs inference locally on full-candidate features, while
DotAegis continues to train the shared classifier. With explicit opt-in the
extension sends new numeric corrections to `/extension/feedback`; after admin
review, the server trains and publishes updated inference weights. Clients
can download those weights without submitting scanner inputs or registering a
device. The extension never submits raw candidates/code to the analysis APIs.
Legacy APIs and historical records remain; there is no blanket zero-storage claim.

A small NumPy transformer classifies 35 numeric properties of a candidate,
its sanitized context, and its variable name. It recognizes patterns; it does
not verify whether a credential is valid, and cannot guarantee detection of
all providers or formats. Raw key text is not a model input or stored training record.

- The release includes `src/models/bootstrap_v2.json.gz`, trained exclusively
  on reproducible synthetic data. Its checksum and independent synthetic
  validation results are in `bootstrap_v2.manifest.json`. Those results are
  not a real-world accuracy guarantee.
- Database checkpoints take priority over the bootstrap. Incompatible older
  token-based checkpoints are skipped. New checkpoints include numeric replay
  samples and Adam state, so reviewed learning survives restarts.
- Inference results are keyed by the actual model-weight revision, preventing
  old cached verdicts after training, reset, or version changes.
- Exact gradients cover attention softmax, residual branches, feed-forward
  layers, LayerNorm, feature embeddings, and classification. Tests compare
  each parameter group against numerical derivatives.

## Live inference releases and client improvement loop

`GET /model/release` is public and read-only. It publishes the **current trained
model**, including approved community learning, as a JSON manifest plus
`weights_base64` (gzip JSON, `dotenvy-local-transformer-v1`). No replay records,
installation data, feedback or optimizer state is exported. The manifest
contains schema 2, SHA-256, weight revision, 35-feature count and label order.
`ETag`/`If-None-Match` return 304 for an unchanged revision. Incompatible or
untrained models return 503. Extension loaders require the supported 64-hidden,
two-layer, four-head architecture and validate shapes/checksums/finite numbers.

The bundled export script still uses reproducible synthetic bootstrap weights
for an offline fallback. Runtime releases use the active trained checkpoint,
not a frozen copy of that bootstrap. DotEnvy checks startup/hourly (independently
switchable), starts replacement workers before switching and clears scan caches.

Community samples remain untrusted until reviewed. An administrator uses
`GET /feedback/pending` and `POST /feedback/review` with
`{"ids":[123],"approve":true}` under admin Bearer authentication. Review triggers
training and durable publication atomically; rejection records the decision
without training. An upload alone does not train the model. Do not mass-approve
unexamined labels; evaluate effects on independent examples. Numeric patterns
can generalize but cannot distinguish every pair of otherwise similar values.

## Privacy and community feedback

In DotEnvy 2.2.3 and earlier, cloud analysis is opt-in. When enabled, candidates and sanitized
context are processed transiently. Request bodies and raw candidate values are
not retained in analysis logs, feedback tables, replay samples, or checkpoints.
Numeric features still describe properties of a value; they should not be
called anonymous or equivalent to storing no data.

`POST /extension/feedback` accepts **numeric features only**. Raw keys and code
fields are rejected. Labels must match the user's action. Each batch contains
1–20 samples, each with a stable ID, feature schema 2, and 35 finite values in
[0, 1]. A registered device may add at most 100 new samples per UTC day; retries
are idempotent. Feedback enters a durable review queue and never changes the
shared model automatically. An administrator reviews samples before approving
training. Do not approve untrusted samples without independent validation.

The full SHA-256 community fingerprint covers `[variableName, completeValue]`.
Version 2 tables isolate old prefix-based entries. Votes stage candidates;
admin-authenticated review must supply matching real evidence and a high model
verdict before promotion. False-positive votes also require admin removal, so newly
registered identities cannot delete trusted entries automatically. A hash alone cannot be classified. The evidence is
not persisted. Community membership describes a reviewed detection pattern,
not proof that a key was leaked or that it is active.

## Authentication

Each installation registers once through `/extension/register` and receives a
random per-device HMAC credential, stored by DotEnvy in VS Code SecretStorage.
Re-registering an existing ID requires a request signed by that device's
current credential. Revoked devices cannot self-reactivate. Lost credentials
require a new installation ID; machine IDs are never ownership proof.

Extension requests use `X-Machine-ID`, `X-Extension-Timestamp`, and
`X-Extension-Signature`. The signature is HMAC-SHA256 over `timestamp + "." + body`.
Timestamps must be finite and within five minutes. This freshness window is not
single-use replay prevention. Shared-secret legacy authentication is disabled.

Admin endpoints require `Authorization: Bearer <API_KEY>`. Optional JWT admin
access requires an explicitly configured secret of at least 32 characters and
an admin token type. Missing JWT configuration disables JWT authentication;
there is no default signing secret.

## Endpoints

| Route | Access | Behavior |
| --- | --- | --- |
| `GET /health`, `/readiness` | Public | 200 only when model is trained and DB responds; otherwise 503 |
| `GET /stats` | Public | Model, training, cache, and aggregate service status |
| `POST /extension/register` | New device / signed rotation | Create or rotate per-device credentials |
| `POST /extension/analyze`, `/extension/analyze/stream` | Device HMAC | Classify a candidate transiently |
| `POST /extension/feedback` | Device HMAC | Queue numeric observations; returns acknowledged sample IDs |
| `GET /feedback/pending`, `POST /feedback/review` | Admin | Inspect, approve, or reject queued observations |
| `POST /analyze`, `/train` | Admin | Analyze / train trusted samples |
| `POST /reset` | Admin | Restore the validated bootstrap |
| `POST /extension/blacklist/add`, `/report_fp`; `GET /extension/blacklist` | Device HMAC | Stage full hashes, report false positives, or sync v2 entries |
| `POST /blacklist/review`, `/blacklist/remove` | Admin | Review matching evidence or remove false positives |
| `POST /cache/clear` | Admin | Delete Aegis cache keys only |

## Run and test

```sh
python -m pip install -r requirements.txt
ENVIRONMENT=development API_KEY=local-admin-key python main.py
# In another terminal:
python -m pip install pytest httpx
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
python scripts/smoke_service.py --url http://localhost:8000
```

The HTTP smoke test uses synthetic candidates and leaves one observation in the
review queue without updating model weights. On an isolated `ENVIRONMENT=test`
service, set `AE_TEST_ADMIN_KEY` and add `--review` to exercise atomic approvals
and idempotent review. Release verification also checks checkpoint restoration
after restarting a PostgreSQL-backed test service.

For a reproducible bootstrap rebuild:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_bootstrap.py --epochs 20
```

Rebuilding writes a release checkpoint, not the production database. Review its
validation manifest and run the regression suite before shipping it.

Export the synthetic bootstrap for an offline extension release:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/export_extension_model.py \
  --output /path/to/dotenvy/resources/models \
  --fixtures /path/to/dotenvy/test/model-parity.json
```

The export verifies the bootstrap checksum, strips replay and optimizer data,
and creates an inference manifest and Python probability fixtures for the
TypeScript runtime. Release model updates through the extension package.

## Railway deployment

Set `ENVIRONMENT=production`, `API_KEY`, and a persistent PostgreSQL
`DATABASE_URL`. Configure `REDIS_URL` using the Redis service reference for
shared caching; when unavailable, inference uses bounded in-process LRU caching.
Redis degradation is visible in `/stats` and does not invalidate the trained model.
Use `WEB_CONCURRENCY=1` while this service owns in-process training/version state.
Plain `postgres://` and `postgresql://` URLs are normalized to the explicitly
installed `psycopg2` driver, independent of SQLAlchemy's default driver choice.

The Docker image bundles the tested bootstrap. On first boot, or after an
incompatible schema upgrade, the service persists that bootstrap to PostgreSQL.
Production refuses missing admin authentication, unavailable/nonpersistent DB,
or an untrained model. Configure Railway's health check as `/readiness`.
Training and feedback status are published atomically to the database; shutdown
never overwrites newer weights with a stale worker snapshot.

Legacy clients submitting feedback or community hashes must use schema-v2
payloads. Earlier clients can continue registered analysis but must update
outdated payloads. DotEnvy **2.2.4+** submits only optional numeric corrections
and public model-release requests; it never calls raw analysis or hash-sync APIs.
Registration rotation without device proof is intentionally rejected.

Licensed under Apache-2.0. See [LICENSE](LICENSE).
