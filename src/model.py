"""
Custom LLM Model Implementation
==============================

Lightweight transformer-based model for secret detection.
Implements a custom transformer architecture optimized for classification.
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional, List
from .attention import CustomAttention
from .feature_extractor import NUM_FEATURES  # ← single source of truth


class ModelConfig:
    """Configuration for the CustomLLM model."""

    def __init__(self):
        self.vocab_size = 1000
        self.hidden_dim = 64
        self.num_layers = 2
        self.num_heads = 4
        self.max_seq_len = 50
        self.num_classes = 4    # high, medium, low, false_positive
        self.dropout = 0.1


class FeedForwardNetwork:
    """Feed-forward network with layer normalization."""

    def __init__(self, hidden_dim: int, ff_dim: Optional[int] = None):
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.w1 = np.random.randn(hidden_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.w2 = np.random.randn(ff_dim, hidden_dim) * 0.02
        self.b2 = np.zeros(hidden_dim)
        self.ln_weight = np.ones(hidden_dim)
        self.ln_bias = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-6)
        x_norm = x_norm * self.ln_weight + self.ln_bias
        h = np.maximum(0, np.matmul(x_norm, self.w1) + self.b1)
        return np.matmul(h, self.w2) + self.b2

    def get_weights(self) -> dict:
        return {
            'w1': self.w1.tolist(), 'b1': self.b1.tolist(),
            'w2': self.w2.tolist(), 'b2': self.b2.tolist(),
            'ln_weight': self.ln_weight.tolist(), 'ln_bias': self.ln_bias.tolist(),
            'hidden_dim': self.hidden_dim, 'ff_dim': self.ff_dim
        }

    def set_weights(self, weights: dict):
        self.w1 = np.array(weights['w1'])
        self.b1 = np.array(weights['b1'])
        self.w2 = np.array(weights['w2'])
        self.b2 = np.array(weights['b2'])
        self.ln_weight = np.array(weights['ln_weight'])
        self.ln_bias = np.array(weights['ln_bias'])
        self.hidden_dim = weights['hidden_dim']
        self.ff_dim = weights['ff_dim']


class TransformerBlock:
    """Single transformer block with attention and feed-forward."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.attention = CustomAttention(config.hidden_dim, config.num_heads)
        self.ffn = FeedForwardNetwork(config.hidden_dim)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn_out = self.attention.forward(x, x, x, mask)
        x = x + attn_out
        ffn_out = self.ffn.forward(x)
        return x + ffn_out

    def get_weights(self) -> dict:
        return {'attention': self.attention.get_weights(), 'ffn': self.ffn.get_weights()}

    def set_weights(self, weights: dict):
        self.attention.set_weights(weights['attention'])
        self.ffn.set_weights(weights['ffn'])


class CustomLLM:
    """Transformer classifier over 35 numeric feature tokens.

    Raw values are used only by the extractor. Both inference and reviewed
    feedback train this same representation; checkpoints contain no keys or code.
    """
    INPUT_MODE = 'features_v2'
    MAX_TRAINING_SAMPLES = 500
    LABELS = ['high', 'medium', 'low', 'false_positive']

    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_embedding = np.random.randn(NUM_FEATURES, config.hidden_dim) * 0.02
        position = np.arange(NUM_FEATURES).reshape(-1, 1)
        div = np.exp(np.arange(0, config.hidden_dim, 2) * -(np.log(10000.0) / config.hidden_dim))
        self.pos_encoding = np.zeros((NUM_FEATURES, config.hidden_dim))
        self.pos_encoding[:, 0::2] = np.sin(position * div)
        self.pos_encoding[:, 1::2] = np.cos(position * div)
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.classifier = np.random.randn(config.hidden_dim, config.num_classes) * 0.02
        self.classifier_bias = np.zeros(config.num_classes)
        self.is_trained = False
        self.learning_rate = 0.001
        self.training_samples = []
        self.confidence_calibration = {}
        self.calibration_samples = []
        self.training_steps = 0
        self.validation = {}
        self._init_adam_state()

    def _features(self, features):
        features = np.asarray(features, dtype=float)
        if features.shape != (NUM_FEATURES,) or not np.isfinite(features).all():
            raise ValueError('Expected 35 finite features')
        if np.any(features < 0) or np.any(features > 1):
            raise ValueError('Features must be between 0 and 1')
        return features

    def _forward(self, features):
        f = self._features(features)
        x = (f[:, None] * self.feature_embedding + self.pos_encoding)[None, :, :]
        caches = []
        for layer in self.layers:
            a, n = layer.attention, layer.ffn
            q, k, v = [a._split_heads(x @ w) for w in (a.w_q, a.w_k, a.w_v)]
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(a.head_dim)
            probs = a._softmax(scores)
            merged = a._concat_heads(probs @ v)
            z = x + merged @ a.w_o
            mean = z.mean(axis=-1, keepdims=True)
            inv_std = 1 / np.sqrt(z.var(axis=-1, keepdims=True) + 1e-6)
            norm = (z - mean) * inv_std
            ln = norm * n.ln_weight + n.ln_bias
            pre = ln @ n.w1 + n.b1
            h = np.maximum(0, pre)
            caches.append((x, q, k, v, probs, merged, inv_std, norm, ln, pre, h))
            x = z + h @ n.w2 + n.b2
        pooled = x.mean(axis=1)
        probs = self._softmax((pooled @ self.classifier + self.classifier_bias)[0])
        return probs, pooled, caches

    def forward(self, secret_value: str, features: np.ndarray) -> Dict[str, Any]:
        probs, _, _ = self._forward(features)
        i = int(np.argmax(probs))
        return {'prediction': self.LABELS[i], 'confidence': float(probs[i]),
                'probabilities': probs.tolist()}

    @staticmethod
    def _softmax(x):
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    def _init_adam_state(self):
        self._adam_t = 0
        self._adam_b1, self._adam_b2, self._adam_eps = 0.9, 0.999, 1e-8
        self._adam_m, self._adam_v = {}, {}

    def _adam_update(self, param, grad, key):
        m = self._adam_m.get(key, np.zeros_like(param))
        v = self._adam_v.get(key, np.zeros_like(param))
        m = self._adam_b1 * m + (1 - self._adam_b1) * grad
        v = self._adam_b2 * v + (1 - self._adam_b2) * grad ** 2
        self._adam_m[key], self._adam_v[key] = m, v
        mh = m / (1 - self._adam_b1 ** self._adam_t)
        vh = v / (1 - self._adam_b2 ** self._adam_t)
        return param - self.learning_rate * mh / (np.sqrt(vh) + self._adam_eps)

    def gradients(self, features, target_label):
        """Exact chain rule, including residuals, LayerNorm and attention softmax."""
        f = self._features(features)
        target = self.LABELS.index(target_label)
        probs, pooled, caches = self._forward(f)
        loss = -np.log(max(probs[target], 1e-30))
        dl = probs.copy()
        dl[target] -= 1
        grads = {'classifier': np.outer(pooled[0], dl), 'classifier_bias': dl}
        dx = np.tile((dl @ self.classifier.T)[None, None, :], (1, NUM_FEATURES, 1)) / NUM_FEATURES
        for i in reversed(range(len(self.layers))):
            a, n = self.layers[i].attention, self.layers[i].ffn
            x, q, k, v, ap, merged, inv_std, norm, ln, pre, h = caches[i]
            flat = lambda value: value.reshape(-1, value.shape[-1])
            prefix = f'layer{i}_ffn_'
            grads[prefix + 'w2'] = flat(h).T @ flat(dx)
            grads[prefix + 'b2'] = dx.sum(axis=(0, 1))
            dh = (dx @ n.w2.T) * (pre > 0)
            grads[prefix + 'w1'] = flat(ln).T @ flat(dh)
            grads[prefix + 'b1'] = dh.sum(axis=(0, 1))
            dln = dh @ n.w1.T
            grads[prefix + 'ln_weight'] = (dln * norm).sum(axis=(0, 1))
            grads[prefix + 'ln_bias'] = dln.sum(axis=(0, 1))
            dn = dln * n.ln_weight
            dz = dx + inv_std * (dn - dn.mean(axis=-1, keepdims=True)
                                - norm * (dn * norm).mean(axis=-1, keepdims=True))
            prefix = f'layer{i}_attn_'
            grads[prefix + 'w_o'] = flat(merged).T @ flat(dz)
            dc = a._split_heads(dz @ a.w_o.T)
            dap = dc @ v.transpose(0, 1, 3, 2)
            dv = ap.transpose(0, 1, 3, 2) @ dc
            ds = ap * (dap - (dap * ap).sum(axis=-1, keepdims=True))
            dq = ds @ k / np.sqrt(a.head_dim)
            dk = ds.transpose(0, 1, 3, 2) @ q / np.sqrt(a.head_dim)
            dx = dz.copy()
            for name, derivative in [('w_q', dq), ('w_k', dk), ('w_v', dv)]:
                derivative = a._concat_heads(derivative)
                grads[prefix + name] = flat(x).T @ flat(derivative)
                dx += derivative @ getattr(a, name).T
        grads['feature_embedding'] = f[:, None] * dx[0]
        return float(loss), grads

    def parameters(self):
        p = {'feature_embedding': self.feature_embedding, 'classifier': self.classifier,
             'classifier_bias': self.classifier_bias}
        for i, layer in enumerate(self.layers):
            for name in ['w1', 'b1', 'w2', 'b2', 'ln_weight', 'ln_bias']:
                p[f'layer{i}_ffn_{name}'] = getattr(layer.ffn, name)
            for name in ['w_q', 'w_k', 'w_v', 'w_o']:
                p[f'layer{i}_attn_{name}'] = getattr(layer.attention, name)
        return p

    def train_step(self, secret_value, features, target_label):
        loss, grads = self.gradients(features, target_label)
        norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        self._adam_t += 1
        for name, param in self.parameters().items():
            param[:] = self._adam_update(param, grads[name] * scale, name)
        self.training_steps += 1
        self.is_trained = True
        return loss

    def train(self, training_samples, epochs=1):
        if not training_samples or epochs < 1:
            raise ValueError('Non-empty samples and positive epochs required')
        losses = []
        for _ in range(epochs):
            for sample in training_samples:
                losses.append(self.train_step('', sample['features'], sample['label']))
        return {'epochs_trained': epochs, 'average_loss': float(np.mean(losses)),
                'samples_processed': len(training_samples), 'model_updated': True}

    def add_training_sample(self, secret_value, features, label):
        if label not in self.LABELS:
            raise ValueError('Invalid training label')
        self.training_samples.append({'features': self._features(features).tolist(), 'label': label})
        self.training_samples = self.training_samples[-self.MAX_TRAINING_SAMPLES:]

    def get_training_stats(self):
        return {'is_trained': self.is_trained, 'training_samples_count': len(self.training_samples),
                'training_steps': self.training_steps, 'learning_rate': self.learning_rate,
                'calibration_samples': len(self.calibration_samples), 'num_features': NUM_FEATURES,
                'validation': self.validation}

    def save_model(self, filepath):
        data = {'config': {'input_mode': self.INPUT_MODE, 'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers, 'num_heads': self.config.num_heads,
                'num_classes': self.config.num_classes, 'num_features': NUM_FEATURES},
                'parameters': {k: v.tolist() for k, v in self.parameters().items()},
                'is_trained': self.is_trained, 'training_samples': self.training_samples,
                'training_steps': self.training_steps, 'validation': self.validation,
                'optimizer': {'step': self._adam_t, 'm': {k:v.tolist() for k,v in self._adam_m.items()},
                              'v': {k:v.tolist() for k,v in self._adam_v.items()}}}
        import tempfile
        directory = os.path.dirname(os.path.abspath(filepath))
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as tmp:
            json.dump(data, tmp, separators=(',', ':'), allow_nan=False)
            temporary = tmp.name
        os.replace(temporary, filepath)

    def load_model(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
        cfg = data['config']
        if cfg.get('input_mode') != self.INPUT_MODE or cfg.get('num_features') != NUM_FEATURES:
            raise ValueError('Checkpoint input schema is incompatible; retraining required')
        config = ModelConfig()
        for key in ['hidden_dim', 'num_layers', 'num_heads', 'num_classes']:
            setattr(config, key, cfg[key])
        candidate = CustomLLM(config)
        for key, param in candidate.parameters().items():
            weights = np.asarray(data['parameters'][key], dtype=float)
            if weights.shape != param.shape or not np.isfinite(weights).all():
                raise ValueError('Invalid checkpoint weights')
            param[:] = weights
        for sample in data.get('training_samples', [])[-self.MAX_TRAINING_SAMPLES:]:
            candidate.add_training_sample('', sample['features'], sample['label'])
        candidate.is_trained = bool(data['is_trained'])
        candidate.training_steps = data.get('training_steps', 0)
        candidate.validation = data.get('validation', {})
        optimizer = data.get('optimizer', {})
        candidate._adam_t = optimizer.get('step', 0)
        for field in ['m', 'v']:
            moments = {}
            for key, value in optimizer.get(field, {}).items():
                array = np.asarray(value)
                if key not in candidate.parameters() or array.shape != candidate.parameters()[key].shape or not np.isfinite(array).all():
                    raise ValueError('Invalid optimizer state')
                moments[key] = array
            setattr(candidate, '_adam_' + field, moments)
        self.__dict__.update(candidate.__dict__)

    def get_model_stats(self):
        return {'hidden_dim': self.config.hidden_dim, 'num_layers': self.config.num_layers,
                'num_features': NUM_FEATURES, 'input_mode': self.INPUT_MODE,
                'is_trained': self.is_trained,
                'model_size_mb': sum(p.nbytes for p in self.parameters().values()) / 1024 ** 2}
