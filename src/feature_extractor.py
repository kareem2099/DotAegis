"""
Enhanced Feature Extractor
==========================

Upgraded from 19 → 35 features for significantly better accuracy.
Organized into 6 feature groups for maintainability.
"""

import re
import math
import string
from typing import Optional
import numpy as np

# ─── Known secret patterns ─────────────────────────────────────────────────────
SECRET_PATTERNS = [
    (r'^sk-[a-zA-Z0-9]{20,}',          1.0,  'stripe_secret'),
    (r'^pk_live_[a-zA-Z0-9]{20,}',     1.0,  'stripe_public'),
    (r'^AKIA[A-Z0-9]{16}',             1.0,  'aws_access_key'),
    (r'^ghp_[a-zA-Z0-9]{36}',          1.0,  'github_pat'),
    (r'^gho_[a-zA-Z0-9]{36}',          1.0,  'github_oauth'),
    (r'^xox[baprs]-[a-zA-Z0-9\-]+',    1.0,  'slack_token'),
    (r'^SG\.[a-zA-Z0-9\-_]{22,}',      1.0,  'sendgrid'),
    (r'^AIza[0-9A-Za-z\-_]{35}',       1.0,  'google_api'),
    (r'^ya29\.[0-9A-Za-z\-_]+',        0.9,  'google_oauth'),
    (r'^eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+$', 1.0, 'jwt'),
    (r'^-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY', 1.0, 'private_key'),
    (r'^[0-9a-f]{32}$',                0.6,  'md5_hash'),
    (r'^[0-9a-f]{40}$',                0.7,  'sha1_hash'),
    (r'^[0-9a-f]{64}$',                0.8,  'sha256_hash'),
    (r'(mysql|postgresql|mongodb|redis|amqp):\/\/\S+:\S+@', 1.0, 'db_url'),
]

HIGH_RISK_CONTEXT = ['password', 'passwd', 'secret', 'private_key', 'api_key',
                     'auth_token', 'access_token', 'client_secret', 'credentials']
MEDIUM_RISK_CONTEXT = ['token', 'key', 'auth', 'api', 'access', 'bearer',
                        'authorization', 'credential', 'config']
LOW_RISK_CONTEXT = ['const', 'let', 'var', 'export', 'process.env',
                     'os.environ', 'getenv', 'dotenv']

NUM_FEATURES = 35  # Update model.py feature_embedding dim to match


def extract(secret: str, context: str, variable_name: Optional[str] = None) -> np.ndarray:
    """
    Extract 35 features from secret + context.

    Feature groups:
      [0-6]   Basic text properties
      [7-13]  Entropy & randomness
      [14-19] Pattern matching
      [20-24] Context risk signals
      [25-29] Variable name signals
      [30-34] Structural analysis
    """
    f = []

    # ── GROUP 1: Basic text (7 features) ──────────────────────────────────────
    length = len(secret)
    f.append(min(1.0, length / 100.0))                                # 0: normalised length
    f.append(1.0 if length >= 20 else length / 20.0)                  # 1: meets min length
    f.append(sum(1 for c in secret if c in string.digits) / max(1, length))   # 2: digit ratio
    f.append(sum(1 for c in secret if c.isupper()) / max(1, length))  # 3: uppercase ratio
    f.append(sum(1 for c in secret if c.islower()) / max(1, length))  # 4: lowercase ratio
    special = set('!#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    f.append(sum(1 for c in secret if c in special) / max(1, length)) # 5: special char ratio
    f.append(len(set(secret)) / max(1, length))                       # 6: unique char ratio

    # ── GROUP 2: Entropy & randomness (7 features) ────────────────────────────
    f.append(_shannon_entropy(secret) / 8.0)                          # 7: normalised entropy
    f.append(_bigram_entropy(secret) / 8.0)                           # 8: bigram entropy
    f.append(_trigram_entropy(secret) / 8.0)                          # 9: trigram entropy
    f.append(_compression_ratio(secret))                              # 10: incompressibility proxy
    f.append(_max_run_length(secret) / max(1, length))                # 11: max repeat run (low = random)
    f.append(1.0 - _max_run_length(secret) / max(1, length))         # 12: randomness score
    f.append(_local_entropy_variance(secret))                         # 13: entropy variance (consistent randomness)

    # ── GROUP 3: Pattern matching (6 features) ────────────────────────────────
    best_pattern_score = 0.0
    matched_pattern = None
    for pattern, score, name in SECRET_PATTERNS:
        if re.search(pattern, secret, re.IGNORECASE | re.MULTILINE):
            if score > best_pattern_score:
                best_pattern_score = score
                matched_pattern = name
    f.append(best_pattern_score)                                       # 14: known pattern match
    f.append(1.0 if _is_base64_like(secret) else 0.0)                 # 15: base64-like
    f.append(1.0 if _is_hex_string(secret) else 0.0)                  # 16: hex string
    f.append(1.0 if secret.startswith(('sk-', 'pk_', 'AKIA', 'ghp_', 'xox', 'SG.', 'AIza', 'ya29.')) else 0.0)  # 17: known prefix
    f.append(1.0 if secret.endswith(('==', '=')) else 0.0)            # 18: base64 padding
    f.append(1.0 if re.search(r'[A-Za-z0-9+/]{40,}={0,2}$', secret) else 0.0)  # 19: long base64

    # ── GROUP 4: Context risk signals (5 features) ───────────────────────────
    ctx = context.lower()
    f.append(min(1.0, sum(0.4 for kw in HIGH_RISK_CONTEXT if kw in ctx)))    # 20: high-risk context
    f.append(min(1.0, sum(0.2 for kw in MEDIUM_RISK_CONTEXT if kw in ctx)))  # 21: medium-risk context
    f.append(min(1.0, sum(0.1 for kw in LOW_RISK_CONTEXT if kw in ctx)))     # 22: assignment context
    f.append(1.0 if any(q in context for q in ('"', "'", '`')) else 0.0)     # 23: quoted value
    f.append(1.0 if '=' in context else 0.0)                                  # 24: assignment operator

    # ── GROUP 5: Variable name signals (5 features) ──────────────────────────
    vn = (variable_name or '').lower()
    f.append(min(1.0, sum(0.5 for kw in ['secret', 'key', 'token', 'password', 'pwd', 'pass'] if kw in vn)))  # 25: dangerous var name
    f.append(min(1.0, sum(0.3 for kw in ['api', 'auth', 'access', 'private', 'cred'] if kw in vn)))           # 26: risk var name
    f.append(1.0 if variable_name and variable_name == variable_name.upper() else 0.0)  # 27: SCREAMING_CASE
    f.append(1.0 if variable_name and '_' in variable_name else 0.0)                     # 28: snake_case
    f.append(0.0 if not variable_name else min(1.0, len(variable_name) / 30.0))          # 29: var name length

    # ── GROUP 6: Structural analysis (5 features) ────────────────────────────
    f.append(_alternating_alpha_digit_score(secret))                   # 30: alternating pattern (API key trait)
    f.append(_separator_structure_score(secret))                       # 31: separator structure (e.g. xxxx-yyyy-zzzz)
    f.append(1.0 if 20 <= length <= 100 else 0.3 if length > 100 else 0.0)  # 32: typical API key length range
    f.append(_character_class_balance(secret))                         # 33: balanced char classes
    f.append(1.0 if re.search(r'[A-Z]{2,}[0-9]{2,}|[0-9]{2,}[A-Z]{2,}', secret) else 0.0)  # 34: uppercase+digit cluster

    assert len(f) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(f)}"
    return np.array(f, dtype=np.float32)


# ─── Helper functions ──────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _ngram_entropy(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    freq = {}
    for g in ngrams:
        freq[g] = freq.get(g, 0) + 1
    total = len(ngrams)
    return -sum((v / total) * math.log2(v / total) for v in freq.values())


def _bigram_entropy(text: str) -> float:
    return _ngram_entropy(text, 2)


def _trigram_entropy(text: str) -> float:
    return _ngram_entropy(text, 3)


def _compression_ratio(text: str) -> float:
    """Estimate incompressibility — high ratio means more random."""
    if not text:
        return 0.0
    unique_chars = len(set(text))
    return min(1.0, unique_chars / min(len(text), 64))


def _max_run_length(text: str) -> int:
    """Length of longest run of repeated characters."""
    if not text:
        return 0
    max_run = cur_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run


def _local_entropy_variance(text: str, window: int = 8) -> float:
    """Low variance = consistently random = likely secret."""
    if len(text) < window:
        return 0.0
    entropies = [_shannon_entropy(text[i:i+window])
                 for i in range(len(text) - window + 1)]
    if not entropies:
        return 0.0
    # Low std dev with high mean → consistent randomness → likely secret
    std = float(np.std(entropies))
    mean = float(np.mean(entropies))
    consistency = 1.0 - min(1.0, std / 2.0)
    level = min(1.0, mean / 4.0)
    return (consistency + level) / 2.0


def _is_base64_like(text: str) -> bool:
    b64_chars = string.ascii_letters + string.digits + '+/='
    return (len(text) % 4 == 0 and
            len(text) >= 16 and
            all(c in b64_chars for c in text))


def _is_hex_string(text: str) -> bool:
    return (len(text) >= 32 and
            all(c in string.hexdigits for c in text) and
            len(text) % 2 == 0)


def _alternating_alpha_digit_score(text: str) -> float:
    """API keys often alternate between alpha and digit characters."""
    if len(text) < 8:
        return 0.0
    switches = sum(
        1 for i in range(len(text) - 1)
        if (text[i].isalpha() and text[i+1].isdigit()) or
           (text[i].isdigit() and text[i+1].isalpha())
    )
    return min(1.0, switches / (len(text) * 0.35))


def _separator_structure_score(text: str) -> float:
    """Tokens like xxxx-yyyy-zzzz are common in API keys."""
    separators = ['-', '_', '.']
    sep_count = sum(text.count(s) for s in separators)
    if sep_count == 0:
        return 0.0
    # Reward consistent segment lengths
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            lengths = [len(p) for p in parts if p]
            if lengths and max(lengths) > 0:
                consistency = 1.0 - (max(lengths) - min(lengths)) / max(lengths)
                return min(1.0, consistency * 0.8 + 0.2)
    return min(1.0, sep_count * 0.2)


def _character_class_balance(text: str) -> float:
    """API keys tend to use all char classes — alpha + digit (+ sometimes special)."""
    if not text:
        return 0.0
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    has_upper = any(c.isupper() for c in text)
    has_lower = any(c.islower() for c in text)
    score = sum([has_alpha, has_digit, has_upper and has_lower]) / 3.0
    return score