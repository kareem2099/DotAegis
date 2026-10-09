"""Reproducible synthetic training; never ingest user keys into a release artifact."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import string
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.model import CustomLLM, ModelConfig
from src.feature_extractor import extract


def dataset(seed, variants=8):
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    text = lambda n: ''.join(rng.choice(alphabet) for _ in range(n))
    samples = []
    for _ in range(variants):
        for prefix, n in [('sk-', 48), ('ghp_', 36), ('xoxb-', 45), ('AIza', 35),
                          ('SG.', 50), ('sk_live_', 40), ('npm_', 40), ('hf_', 40)]:
            value = prefix + text(n)
            context = 'const API_KEY = "[REDACTED]";'
            samples.append((value, context, 'API_KEY', 'high'))
        samples.append(('AKIA' + ''.join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16)),
                        'AWS_ACCESS_KEY_ID="[REDACTED]"', 'AWS_ACCESS_KEY_ID', 'high'))
        for name in ['PASSWORD', 'CLIENT_SECRET', 'ACCESS_TOKEN', 'AUTH_TOKEN']:
            samples.append((text(32), f'const {name} = "[REDACTED]";', name, 'high'))
        for name in ['BUILD_ID', 'CHECKSUM', 'ASSET_HASH', 'VERSION_ID']:
            samples.append((text(32), f'const {name} = "[REDACTED]";', name, 'false_positive'))
        for value in ['hello-world', 'production', 'localhost', 'application/json', 'v2.3.1',
                      'http://localhost:3000', 'user@example.com', 'example-placeholder']:
            samples.append((value, 'const greeting = "[REDACTED]";', 'greeting', 'low'))
        # Public keys are identifiers, not secret credentials.
        samples.append(('pk_live_' + text(40), 'STRIPE_PUBLIC_KEY="[REDACTED]"', 'STRIPE_PUBLIC_KEY', 'false_positive'))
        samples.append((text(32), 'const session = "[REDACTED]";', 'session', 'medium'))
    return samples


def evaluate(model, samples):
    correct = 0
    counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for value, context, name, label in samples:
        pred = model.forward('', extract(value, context, name))['prediction']
        expected = label in ('high', 'medium')
        actual = pred in ('high', 'medium')
        correct += expected == actual
        counts[('tp' if actual else 'fn') if expected else ('fp' if actual else 'tn')] += 1
    return {'samples': len(samples), 'binary_accuracy': correct / len(samples),
            'precision': counts['tp'] / max(1, counts['tp'] + counts['fp']),
            'recall': counts['tp'] / max(1, counts['tp'] + counts['fn']), **counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    np.random.seed(2099)
    model = CustomLLM(ModelConfig())
    train, validation = dataset(2099), dataset(9898, 3)
    for value, context, name, label in train:
        model.add_training_sample('', extract(value, context, name), label)
    rng = random.Random(2099)
    for epoch in range(args.epochs):
        samples = list(model.training_samples)
        rng.shuffle(samples)
        result = model.train(samples)
        print(f'Epoch {epoch+1}: loss={result["average_loss"]:.4f}', flush=True)
    model.validation = evaluate(model, validation)
    model.validation['dataset'] = 'synthetic-independent-seed-v2; not a real-world accuracy guarantee'
    if model.validation['binary_accuracy'] < .90 or model.validation['precision'] < .90 or model.validation['recall'] < .90:
        raise RuntimeError('Independent synthetic validation failed release threshold')
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory) / 'model.json'
        model.save_model(str(tmp))
        payload = tmp.read_bytes()
    destination = ROOT / 'src/models/bootstrap_v2.json.gz'
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(gzip.compress(payload, mtime=0))
    manifest = {'input_mode': model.INPUT_MODE, 'feature_schema': 2,
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'training_samples': len(train), 'epochs': args.epochs, 'validation': model.validation}
    (ROOT / 'src/models/bootstrap_v2.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
