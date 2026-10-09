"""Export synthetic bootstrap inference weights for offline DotEnvy releases."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import CustomLLM, ModelConfig


def export(output):
    source = ROOT / 'src/models/bootstrap_v2.json.gz'
    manifest = json.loads(source.with_name('bootstrap_v2.manifest.json').read_text())
    compressed = source.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest['sha256']:
        raise ValueError('Bootstrap checksum mismatch')
    checkpoint = json.loads(gzip.decompress(compressed))
    if not checkpoint['is_trained']:
        raise ValueError('Cannot export an untrained model')
    payload = {'format': 'dotenvy-local-transformer-v1', 'config': checkpoint['config'],
               'parameters': checkpoint['parameters'], 'is_trained': True}
    data = gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'aegis-v2.json.gz').write_bytes(data)
    release = {'format': payload['format'], 'feature_schema': 2, 'sha256': hashlib.sha256(data).hexdigest(),
               'bootstrap_sha256': manifest['sha256'], 'training_source': 'synthetic-only',
               'validation': checkpoint['validation'], 'num_features': 35,
               'labels': CustomLLM.LABELS}
    (output / 'aegis-v2.manifest.json').write_text(json.dumps(release, indent=2) + '\n')
    # Independent fixtures prove the JS port matches Python on all probabilities.
    with tempfile.NamedTemporaryFile(mode='w') as tmp:
        json.dump(checkpoint, tmp)
        tmp.flush()
        model = CustomLLM(ModelConfig())
        model.load_model(tmp.name)
    feature_cases = json.loads((ROOT / 'tests/feature-parity.json').read_text())
    vectors = [case['features'] for case in feature_cases]
    import numpy as np
    rng = np.random.default_rng(5401)
    vectors += [[0.0]*35, [1.0]*35] + rng.uniform(size=(12, 35)).tolist()
    fixtures = [{'features': f, **model.forward('', f)} for f in vectors]
    return fixtures, len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    fixtures, size = export(args.output)
    args.fixtures.write_text(json.dumps(fixtures, indent=2) + '\n')
    print(f'Exported {size} bytes of inference weights and {len(fixtures)} Python parity cases; no replay or optimizer data.')
