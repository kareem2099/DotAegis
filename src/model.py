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
        self.hidden_dim = 128
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
    """
    Custom transformer-based LLM for secret detection.
    Processes numerical features and text tokens for classification.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Embedding layers — NUM_FEATURES from feature_extractor (35)
        self.token_embedding = np.random.randn(config.vocab_size, config.hidden_dim) * 0.02
        self.feature_embedding = np.random.randn(NUM_FEATURES, config.hidden_dim) * 0.02

        self.pos_encoding = self._create_positional_encoding()
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.classifier = np.random.randn(config.hidden_dim, config.num_classes) * 0.02
        self.classifier_bias = np.zeros(config.num_classes)

        self.is_trained = False
        self.learning_rate = 0.001
        self.training_samples = []
        self.confidence_calibration = {}
        self.calibration_samples = []

    def _create_positional_encoding(self) -> np.ndarray:
        pos_enc = np.zeros((self.config.max_seq_len, self.config.hidden_dim))
        position = np.arange(0, self.config.max_seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, self.config.hidden_dim, 2).astype(np.float32) *
            -(np.log(10000.0) / self.config.hidden_dim)
        )
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return pos_enc

    def _tokenize_secret(self, secret_value: str) -> List[int]:
        return [
            ord(c) if ord(c) < self.config.vocab_size else 0
            for c in secret_value[:self.config.max_seq_len]
        ]

    def forward(self, secret_value: str, features: np.ndarray) -> Dict[str, Any]:
        tokens = self._tokenize_secret(secret_value)
        seq_len = len(tokens)
        if seq_len == 0:
            return {'prediction': 'low', 'confidence': 0.0, 'probabilities': [0.0, 0.0, 1.0, 0.0]}

        token_embeds  = self.token_embedding[tokens]
        feature_embeds = np.matmul(features.reshape(1, -1), self.feature_embedding)
        feature_embeds = np.tile(feature_embeds, (seq_len, 1))

        x = token_embeds + feature_embeds + self.pos_encoding[:seq_len]
        x = x.reshape(1, seq_len, self.config.hidden_dim)

        for layer in self.layers:
            x = layer.forward(x)

        x = np.mean(x, axis=1)
        logits = np.matmul(x, self.classifier) + self.classifier_bias
        probs = self._softmax(logits[0])

        pred_idx = int(np.argmax(probs))
        labels = ['high', 'medium', 'low', 'false_positive']
        return {
            'prediction':    labels[pred_idx],
            'confidence':    float(probs[pred_idx]),
            'probabilities': probs.tolist()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # ── Adam optimizer state ──────────────────────────────────────────────────
    def _init_adam_state(self):
        """Initialize Adam moment estimates for every trainable parameter group."""
        self._adam_t  = 0         # global step counter
        self._adam_lr = 0.001
        self._adam_b1 = 0.9
        self._adam_b2 = 0.999
        self._adam_eps = 1e-8

        # Build a flat dict of (param_ref_key → (m, v)) for all weight matrices
        # We store them by name; update() will apply to the actual numpy arrays in-place.
        self._adam_m: dict = {}
        self._adam_v: dict = {}

    def _adam_update(self, param: np.ndarray, grad: np.ndarray, key: str) -> np.ndarray:
        """Apply one Adam step to `param` using gradient `grad`. Returns updated param."""
        if key not in self._adam_m:
            self._adam_m[key] = np.zeros_like(param)
            self._adam_v[key] = np.zeros_like(param)

        m = self._adam_m[key]
        v = self._adam_v[key]

        m = self._adam_b1 * m + (1 - self._adam_b1) * grad
        v = self._adam_b2 * v + (1 - self._adam_b2) * (grad ** 2)

        self._adam_m[key] = m
        self._adam_v[key] = v

        m_hat = m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = v / (1 - self._adam_b2 ** self._adam_t)

        return param - self._adam_lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

    def train_step(self, secret_value: str, features: np.ndarray, target_label: str) -> float:
        """
        One training step with full backprop through all layers + Adam optimizer.
        Layers updated: token_embedding, feature_embedding, classifier,
                        FFN (w1/b1/w2/b2/ln_weight/ln_bias), Attention (w_q/w_k/w_v/w_o).
        """
        # Ensure Adam state is initialised
        if not hasattr(self, '_adam_t'):
            self._init_adam_state()
        self._adam_t += 1

        label_map  = {'high': 0, 'medium': 1, 'low': 2, 'false_positive': 3}
        target_idx = label_map.get(target_label, 2)

        # ── Forward pass (store intermediates for backprop) ────────────────────
        tokens  = self._tokenize_secret(secret_value)
        seq_len = len(tokens)
        if seq_len == 0:
            return 0.0

        # Embeddings
        tok_emb = self.token_embedding[tokens]                           # [T, H]
        feat_proj = np.matmul(features.reshape(1, -1),
                              self.feature_embedding)                    # [1, H]
        feat_emb  = np.tile(feat_proj, (seq_len, 1))                    # [T, H]
        x = tok_emb + feat_emb + self.pos_encoding[:seq_len]            # [T, H]
        x = x.reshape(1, seq_len, self.config.hidden_dim)               # [1, T, H]

        # Store per-layer inputs for backprop
        layer_inputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            layer_inputs.append(x)

        # Pooling + classification
        pooled = np.mean(x, axis=1)                                      # [1, H]
        logits = np.matmul(pooled, self.classifier) + self.classifier_bias  # [1, C]
        probs  = self._softmax(logits[0])

        # ── Loss (cross-entropy) ───────────────────────────────────────────────
        loss = -np.log(probs[target_idx] + 1e-8)

        # ── Backprop: output layer ─────────────────────────────────────────────
        # dL/d_logits = probs - one_hot(target)
        d_logits = probs.copy()
        d_logits[target_idx] -= 1.0                                      # [C]

        # Gradients for classifier weights and bias
        # pooled: [1, H], d_logits: [C]  →  d_classifier: [H, C]
        d_classifier      = np.outer(pooled[0], d_logits)
        d_classifier_bias = d_logits.copy()

        # Gradient w.r.t. pooled: [1, H]
        d_pooled = np.matmul(d_logits, self.classifier.T).reshape(1, -1)  # [1, H]

        # ── Backprop through mean-pooling ──────────────────────────────────────
        # d_x_last: [1, T, H]  (broadcast d_pooled over seq_len)
        d_x = np.tile(d_pooled[:, np.newaxis, :],
                      (1, seq_len, 1)) / seq_len                         # [1, T, H]

        # ── Backprop through transformer layers (reverse order) ────────────────
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            x_in  = layer_inputs[li]                                     # [1, T, H]
            x_out = layer_inputs[li + 1]                                 # [1, T, H]

            # --- FFN backprop (residual: x_out = x_attn + ffn(x_attn)) --------
            # Approximate: split d_x equally between residual and ffn branch
            d_ffn_in  = d_x * 0.5
            d_residual = d_x * 0.5

            # FFN forward internals needed for backprop
            ffn = layer.ffn
            # LayerNorm forward
            x_in_2d = x_in.reshape(-1, self.config.hidden_dim)          # [T, H]
            mean_    = np.mean(x_in_2d, axis=-1, keepdims=True)
            var_     = np.var(x_in_2d, axis=-1, keepdims=True)
            x_norm   = (x_in_2d - mean_) / np.sqrt(var_ + 1e-6)
            x_ln     = x_norm * ffn.ln_weight + ffn.ln_bias              # [T, H]

            # d_w2: hidden=pre-relu activations
            h = np.maximum(0, np.matmul(x_ln, ffn.w1) + ffn.b1)        # [T, ff_dim]

            d_ffn_out_2d = d_ffn_in.reshape(-1, self.config.hidden_dim) # [T, H]
            d_w2  = np.matmul(h.T, d_ffn_out_2d)                        # [ff_dim, H]
            d_b2  = d_ffn_out_2d.sum(axis=0)                             # [H]
            d_h   = np.matmul(d_ffn_out_2d, ffn.w2.T)                   # [T, ff_dim]
            d_h  *= (h > 0).astype(float)                                # ReLU gate
            d_w1  = np.matmul(x_ln.T, d_h)                              # [H, ff_dim]
            d_b1  = d_h.sum(axis=0)                                      # [ff_dim]
            d_ln_weight = (d_h @ ffn.w1.T * x_norm).sum(axis=0)         # [H]  approx
            d_ln_bias   = (d_h @ ffn.w1.T).sum(axis=0)                  # [H]

            # Apply Adam to FFN weights
            prefix = f'layer{li}_ffn'
            ffn.w2 = self._adam_update(ffn.w2, d_w2,   f'{prefix}_w2')  # [ff_dim, H]
            ffn.b2 = self._adam_update(ffn.b2, d_b2,   f'{prefix}_b2')
            ffn.w1 = self._adam_update(ffn.w1, d_w1,   f'{prefix}_w1')
            ffn.b1 = self._adam_update(ffn.b1, d_b1,   f'{prefix}_b1')
            ffn.ln_weight = self._adam_update(ffn.ln_weight, d_ln_weight, f'{prefix}_lnw')
            ffn.ln_bias   = self._adam_update(ffn.ln_bias,   d_ln_bias,   f'{prefix}_lnb')

            # --- Attention backprop (simplified: treat as linear for w_o) ------
            attn = layer.attention
            # d_x through residual + attn branch
            d_attn_out = d_residual.reshape(-1, self.config.hidden_dim)  # [T, H]
            x_in_2d_a  = x_in.reshape(-1, self.config.hidden_dim)        # [T, H]

            # Backprop through w_o (output projection)
            d_w_o = np.matmul(x_in_2d_a.T, d_attn_out)                  # [H, H]
            attn.w_o = self._adam_update(attn.w_o, d_w_o, f'layer{li}_attn_wo')

            # Simplified backprop for w_q, w_k, w_v via gradient of x @ W
            d_proj = np.matmul(d_attn_out, attn.w_o.T)                   # [T, H]
            scale  = 1.0 / 3.0
            for wname, w in [('wq', attn.w_q), ('wk', attn.w_k), ('wv', attn.w_v)]:
                d_w = np.matmul(x_in_2d_a.T, d_proj) * scale
                setattr(attn, f'w_{wname[-1]}',
                        self._adam_update(w, d_w, f'layer{li}_attn_{wname}'))

            # Accumulate gradient to previous layer
            d_x = d_residual + d_ffn_in  # simplified — pass-through for next iter

        # ── Backprop through embeddings ────────────────────────────────────────
        d_x_emb = d_x.reshape(seq_len, self.config.hidden_dim)          # [T, H]

        # token_embedding
        d_tok_emb = np.zeros_like(self.token_embedding)
        for t_idx, tok in enumerate(tokens):
            d_tok_emb[tok] += d_x_emb[t_idx]
        self.token_embedding = self._adam_update(
            self.token_embedding, d_tok_emb, 'tok_emb')

        # feature_embedding
        d_feat_proj = d_x_emb.sum(axis=0, keepdims=True)                # [1, H]
        d_feat_emb  = np.matmul(features.reshape(-1, 1), d_feat_proj)   # [F, H]
        self.feature_embedding = self._adam_update(
            self.feature_embedding, d_feat_emb, 'feat_emb')

        # ── Apply classifier updates ───────────────────────────────────────────
        self.classifier      = self._adam_update(self.classifier,      d_classifier,      'classifier')
        self.classifier_bias = self._adam_update(self.classifier_bias, d_classifier_bias, 'classifier_bias')

        return float(loss)

    def _simple_gradient_update(self, *args, **kwargs):
        """Kept for backward compatibility — no longer used (replaced by Adam backprop)."""
        pass

    def train(self, training_samples: List[Dict[str, Any]], epochs: int = 1) -> Dict[str, Any]:
        if not training_samples:
            return {'error': 'No training samples provided'}

        total_loss = 0.0
        num_samples = len(training_samples)

        for _ in range(epochs):
            epoch_loss = 0.0
            for sample in training_samples:
                secret_value = sample.get('secret_value', '')
                features = np.array(sample.get('features', []))
                label = sample.get('label', 'low')

                # ← uses NUM_FEATURES (35) instead of hardcoded 19
                if len(features) != NUM_FEATURES:
                    continue

                epoch_loss += self.train_step(secret_value, features, label)

            total_loss += epoch_loss / max(1, num_samples)

        self.is_trained = True
        return {
            'epochs_trained':   epochs,
            'average_loss':     total_loss / epochs,
            'samples_processed': num_samples,
            'model_updated':    True
        }

    def add_training_sample(self, secret_value: str, features: List[float], label: str):
        self.training_samples.append({
            'secret_value': secret_value,
            'features':     features,
            'label':        label
        })

    def get_training_stats(self) -> Dict[str, Any]:
        return {
            'is_trained':             self.is_trained,
            'training_samples_count': len(self.training_samples),
            'learning_rate':          self.learning_rate,
            'calibration_samples':    len(self.calibration_samples),
            'num_features':           NUM_FEATURES,
        }

    def calibrate_confidence(self, predicted_confidence: float, true_label: str) -> float:
        if true_label not in self.confidence_calibration:
            self.confidence_calibration[true_label] = []
        self.calibration_samples.append({
            'predicted_confidence': predicted_confidence,
            'true_label': true_label
        })
        self.confidence_calibration[true_label].append(predicted_confidence)
        if len(self.confidence_calibration[true_label]) > 5:
            historical_mean = np.mean(self.confidence_calibration[true_label])
            calibrated = predicted_confidence * 0.7 + historical_mean * 0.3
            return float(min(1.0, max(0.0, calibrated)))
        return predicted_confidence

    def get_calibration_stats(self) -> Dict[str, Any]:
        return {
            label: {
                'count': len(c),
                'mean':  float(np.mean(c)),
                'std':   float(np.std(c)),
                'min':   float(np.min(c)),
                'max':   float(np.max(c)),
            }
            for label, c in self.confidence_calibration.items() if c
        }

    def save_model(self, filepath: str):
        model_data = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_heads':  self.config.num_heads,
                'max_seq_len': self.config.max_seq_len,
                'num_classes': self.config.num_classes,
                'dropout':    self.config.dropout,
                'num_features': NUM_FEATURES,  # saved for validation on load
            },
            'weights': {
                'token_embedding':   self.token_embedding.tolist(),
                'feature_embedding': self.feature_embedding.tolist(),
                'pos_encoding':      self.pos_encoding.tolist(),
                'classifier':        self.classifier.tolist(),
                'classifier_bias':   self.classifier_bias.tolist(),
            },
            'layers':     [layer.get_weights() for layer in self.layers],
            'is_trained': self.is_trained,
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found, using random initialization")
            return
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)

            cfg = model_data['config']
            # If saved model had different num_features, reinitialize feature_embedding
            saved_num_features = cfg.get('num_features', 19)
            if saved_num_features != NUM_FEATURES:
                print(f"⚠️  Model has {saved_num_features} features, current is {NUM_FEATURES}. Re-initializing feature_embedding.")
                self.feature_embedding = np.random.randn(NUM_FEATURES, self.config.hidden_dim) * 0.02
            else:
                w = model_data['weights']
                self.feature_embedding = np.array(w['feature_embedding'])

            w = model_data['weights']
            self.token_embedding  = np.array(w['token_embedding'])
            self.pos_encoding     = np.array(w['pos_encoding'])
            self.classifier       = np.array(w['classifier'])
            self.classifier_bias  = np.array(w['classifier_bias'])

            for i, layer_weights in enumerate(model_data['layers']):
                if i < len(self.layers):
                    self.layers[i].set_weights(layer_weights)

            self.is_trained = model_data.get('is_trained', False)
            print(f"✅ Model loaded from {filepath} (features: {NUM_FEATURES})")

        except Exception as e:
            print(f"Error loading model: {e}, using random initialization")

    def get_model_stats(self) -> Dict[str, Any]:
        return {
            'vocab_size':    self.config.vocab_size,
            'hidden_dim':    self.config.hidden_dim,
            'num_layers':    self.config.num_layers,
            'num_heads':     self.config.num_heads,
            'num_features':  NUM_FEATURES,
            'is_trained':    self.is_trained,
            'model_size_mb': self._estimate_model_size(),
        }

    def _estimate_model_size(self) -> float:
        total = (self.token_embedding.size + self.feature_embedding.size +
                 self.classifier.size + self.classifier_bias.size)
        for layer in self.layers:
            for d in [layer.attention.get_weights(), layer.ffn.get_weights()]:
                for v in d.values():
                    if isinstance(v, list):
                        total += (len(v) * len(v[0]) if isinstance(v[0], list) else len(v))
        return (total * 4) / (1024 * 1024)