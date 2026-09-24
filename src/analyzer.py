"""LLMAnalyzer — wraps CustomLLM with feature extraction, caching, versioning."""
import os
import re
import math
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np

from .model import CustomLLM, ModelConfig
from .feature_extractor import extract as extract_features
from .cache_manager import EnhancedCacheManager


# Module-level cache instance (shared across all requests in the process)
enhanced_cache = EnhancedCacheManager()


class LLMAnalyzer:
    def __init__(self):
        self.config = ModelConfig()
        self.model  = CustomLLM(self.config)
        self.load_model()

        self.cache_hits   = 0
        self.cache_misses = 0

        self.model_versions      = {}
        self.active_version      = "default"
        self.ab_testing_enabled  = False
        self.ab_test_groups      = {"A": "default", "B": "default"}
        self.version_performance = {}

    # ── Persistence (T05: DB-primary, file-fallback) ──────────────────────────
    def load_model(self):
        """
        Load model weights. Priority:
          1. Database (survives Railway redeploys)
          2. Local JSON file (development / first-boot)
        """
        # 1. Try database
        try:
            from .database import db_manager
            json_str = db_manager.load_model_from_db()
            if json_str:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                tmp.write(json_str)
                tmp.close()
                self.model.load_model(tmp.name)
                os.unlink(tmp.name)
                return
        except Exception as e:
            print(f"⚠️  DB model load failed, trying local file: {e}")

        # 2. Fallback: local JSON file
        path = os.path.join(os.path.dirname(__file__), "models", "llm_model.json")
        if os.path.exists(path):
            self.model.load_model(path)

    def save_model(self):
        """
        Save model weights to both DB (primary) and local JSON (fallback).
        """
        import json as _json
        path = os.path.join(os.path.dirname(__file__), "models", "llm_model.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write to local file first (always works, needed for dev)
        self.model.save_model(path)

        # Then persist to DB (survives redeploys)
        try:
            from .database import db_manager
            with open(path, 'r') as f:
                weights_json = f.read()
            num_samples = len(self.model.training_samples)
            db_manager.save_model_to_db(weights_json, version="active",
                                        num_samples=num_samples)
        except Exception as e:
            print(f"⚠️  DB model save failed (local file still saved): {e}")

    # ── Feature extraction ────────────────────────────────────────────────────
    def extract_features(self, secret_value: str, context: str,
                          variable_name: Optional[str] = None) -> np.ndarray:
        return extract_features(secret_value, context, variable_name)

    # ── Confidence with two-tier cache ────────────────────────────────────────
    def calculate_enhanced_confidence(self, secret_value: str, context: str,
                                       traditional_confidence: str,
                                       variable_name: Optional[str] = None) -> str:
        key    = enhanced_cache.make_key(secret_value, context, variable_name)
        cached = enhanced_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        features = self.extract_features(secret_value, context, variable_name)
        result   = self.model.forward(secret_value, features)

        pred = result.get('prediction', 'low')
        if pred == 'high':
            level = "high"
        elif pred == 'medium':
            level = "medium"
        elif pred in ('low', 'false_positive'):
            level = "low"
        else:
            level = traditional_confidence

        enhanced_cache.set(key, level, ttl=enhanced_cache.pick_ttl(level))
        return level

    def analyze(self, secret_value: str, context: str,
                variable_name: Optional[str] = None) -> Dict[str, Any]:
        level = self.calculate_enhanced_confidence(secret_value, context, "low", variable_name)
        return {
            "enhanced_confidence": level,
            "category": self._categorize_secret(secret_value),
            "risk_level": self._assess_risk_level(secret_value, context),
            "is_likely_secret": level in ("high", "medium"),
        }

    def record_user_feedback(self, secret_value: str, context: str,
                              features: List[float], user_action: str,
                              variable_name: Optional[str] = None):
        """Record user feedback, persist sample to DB, and update model weights."""
        label_map = {
            "confirmed_secret": "high",
            "marked_false_positive": "false_positive",
            "ignored_warning": "medium",
        }
        label = label_map.get(user_action, "false_positive" if "false" in user_action else "high")
        try:
            from .database import db_manager
            db_manager.store_training_sample(
                secret_hash=hashlib.sha256(secret_value.encode()).hexdigest(),
                context_hash=hashlib.sha256(context.encode()).hexdigest(),
                features=features,
                label=label,
                user_action=user_action,
                confidence=0.5,
                model_version=self.active_version,
            )
            self.model.add_training_sample(secret_value, features, label)
            feat_arr = np.array(features, dtype=float)
            self.model.train_step(secret_value, feat_arr, label)
            self.save_model()
        except Exception as e:
            print(f"⚠️ Error recording user feedback: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return self.model.get_model_stats()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq: Dict[str, int] = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        return -sum((v / len(text)) * math.log2(v / len(text)) for v in freq.values())

    def _categorize_secret(self, s: str) -> str:
        if re.match(r'^(sk|pk)[_-]', s):                          return 'API Key'
        if re.match(r'^(AKIAI|AKIAIOS)', s):                       return 'AWS API Key'
        if re.match(r'^ghp_', s):                                  return 'GitHub Token'
        if re.match(r'^xox[bap]-', s):                            return 'Slack Token'
        if re.match(r'^SG\.', s):                                  return 'SendGrid API Key'
        if re.match(r'^(mysql|postgresql|mongodb|redis):\/\/', s): return 'Database URL'
        if re.match(r'^Bearer\s+', s):                             return 'Bearer Token'
        if re.match(r'-----BEGIN', s):                             return 'Certificate/Private Key'
        if re.match(r'^eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.', s): return 'JWT Token'
        if len(s) >= 64:                                           return 'Cryptographic Key'
        return 'Potential Secret'

    def _assess_risk_level(self, secret_value: str, context: str) -> str:
        ctx = context.lower()
        if 'authorization' in ctx or 'bearer' in ctx: return 'critical'
        if 'stripe' in ctx or 'payment' in ctx:       return 'high'
        if 'aws' in ctx or 'cloud' in ctx:            return 'high'
        if 'api' in ctx or 'key' in ctx or 'token' in ctx: return 'medium'
        return 'low'

    # ── Model versioning ──────────────────────────────────────────────────────
    def create_model_version(self, version_name: str) -> bool:
        if version_name in self.model_versions:
            return False
        try:
            v = CustomLLM(self.config)
            v.token_embedding   = self.model.token_embedding.copy()
            v.feature_embedding = self.model.feature_embedding.copy()
            v.pos_encoding      = self.model.pos_encoding.copy()
            v.classifier        = self.model.classifier.copy()
            v.classifier_bias   = self.model.classifier_bias.copy()
            v.layers            = self.model.layers.copy()
            self.model_versions[version_name] = v
            self.version_performance[version_name] = {
                'total_predictions': 0, 'correct_predictions': 0,
                'accuracy': 0.0, 'created_at': 'now'
            }
            return True
        except Exception as e:
            print(f"Error creating version: {e}")
            return False

    def switch_model_version(self, version_name: str) -> bool:
        if version_name not in self.model_versions and version_name != "default":
            return False
        self.active_version = version_name
        if version_name != "default":
            self.model = self.model_versions[version_name]
        self.cache_hits = self.cache_misses = 0
        return True

    def enable_ab_testing(self, version_a="default", version_b="default") -> bool:
        if version_a not in self.model_versions and version_a != "default": return False
        if version_b not in self.model_versions and version_b != "default": return False
        self.ab_testing_enabled = True
        self.ab_test_groups = {"A": version_a, "B": version_b}
        return True

    def disable_ab_testing(self):
        self.ab_testing_enabled = False
        self.switch_model_version("default")

    def get_ab_test_results(self) -> Dict[str, Any]:
        if not self.ab_testing_enabled:
            return {"error": "A/B testing not enabled"}
        results = {g: {'version': v, 'performance': self.version_performance.get(v,
                   {'total_predictions': 0, 'accuracy': 0.0})}
                   for g, v in self.ab_test_groups.items()}
        pa, pb = results['A']['performance'], results['B']['performance']
        if pa['total_predictions'] > 10 and pb['total_predictions'] > 10:
            winner = 'A' if pa['accuracy'] > pb['accuracy'] else \
                     'B' if pb['accuracy'] > pa['accuracy'] else 'tie'
        else:
            winner = 'insufficient_data'
        results['winner'] = winner
        results['recommendation'] = self.ab_test_groups.get(winner)
        return results

    def record_prediction_result(self, version: str, correct: bool):
        p = self.version_performance.setdefault(version,
            {'total_predictions': 0, 'correct_predictions': 0, 'accuracy': 0.0})
        p['total_predictions'] += 1
        if correct:
            p['correct_predictions'] += 1
        p['accuracy'] = p['correct_predictions'] / p['total_predictions']

    def get_model_version_info(self) -> Dict[str, Any]:
        return {
            'active_version':     self.active_version,
            'ab_testing_enabled': self.ab_testing_enabled,
            'versions': {
                'default': {'exists': True, 'performance': self.version_performance.get(
                    'default', {'total_predictions': 0, 'accuracy': 0.0})},
                **{n: {'exists': True, 'performance': self.version_performance.get(
                    n, {'total_predictions': 0, 'accuracy': 0.0})}
                   for n in self.model_versions}
            }
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
analyzer = LLMAnalyzer()
