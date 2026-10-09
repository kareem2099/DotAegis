"""Regression coverage for registration, privacy, exact gradients and persistence."""
import asyncio
import gzip
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import time

os.environ['ENVIRONMENT'] = 'test'
os.environ['DATABASE_URL'] = 'sqlite:///' + tempfile.mktemp(suffix='.db')
os.environ['REDIS_URL'] = ''
os.environ['API_KEY'] = 'synthetic-admin-test-key'

import numpy as np
import pytest
import jwt
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request
from src.analyzer import analyzer, enhanced_cache
from src.database import db_manager, ExtensionClient, CommunityFeedback
from src.extension_auth import _client_credentials_cache, _rate_limit_store, _ip_registration_store, verify_extension_signature, get_cached_client_credentials
from src.feature_extractor import extract
from src.model import CustomLLM, ModelConfig
from src.models import ExtensionRegisterRequest, FeedbackRequest, FeedbackReviewRequest, HashSubmitRequest, BlacklistReviewRequest
from src.routes.analyze import extension_register, extension_feedback, blacklist_add, review_blacklist, _build_response
from src.routes.stats import health
from src.routes.train import review_feedback
from src.security import SecurityManager


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    if db_manager.engine:
        db_manager.engine.dispose()
    db_manager.database_url = 'sqlite:///' + str(tmp_path / 'review.db')
    db_manager.engine = db_manager.SessionLocal = None
    db_manager.initialize()
    _client_credentials_cache.clear()
    _rate_limit_store.clear()
    _ip_registration_store.clear()
    enhanced_cache.clear()
    np.random.seed(2099)
    config = ModelConfig()
    config.hidden_dim, config.num_layers, config.num_heads = 8, 1, 2
    analyzer.model = CustomLLM(config)
    analyzer.model.is_trained = True
    analyzer._default_model = analyzer.model
    analyzer.active_version = 'default'
    analyzer.revision = analyzer._model_revision()
    analyzer.model_versions = {}
    def save_model(review_ids=None):
        path = tmp_path / 'model.json'
        analyzer.model.save_model(str(path))
        assert db_manager.save_model_to_db(path.read_text(), review_ids=review_ids)
        analyzer.revision = analyzer._model_revision()
    monkeypatch.setattr(analyzer, 'save_model', save_model)


def request(body, secret=None, machine='review-device-123'):
    payload = json.dumps(body).encode()
    ts = str(time.time())
    headers = []
    if secret:
        signature = hmac.new(secret.encode(), ts.encode() + b'.' + payload, hashlib.sha256).hexdigest()
        headers = [(b'x-machine-id', machine.encode()), (b'x-extension-timestamp', ts.encode()),
                   (b'x-extension-signature', signature.encode())]
    req = Request({'type': 'http', 'client': ('127.0.0.1', 1), 'headers': headers})
    req._body = payload
    return req


def register(machine='review-device-123', secret=None):
    body = {'machine_id': machine}
    return asyncio.run(extension_register(ExtensionRegisterRequest(**body), request(body, secret, machine)))


def test_registration_requires_proof_and_cannot_reactivate():
    first = register()
    with pytest.raises(HTTPException) as exc:
        register()
    assert exc.value.status_code == 401
    rotated = register(secret=first.client_secret)
    assert rotated.client_secret != first.client_secret
    with db_manager.get_session() as session:
        device = session.query(ExtensionClient).first()
        device.is_active = False
        session.commit()
    with pytest.raises(HTTPException) as exc:
        register(secret=rotated.client_secret)
    assert exc.value.status_code == 403
    assert not db_manager.get_client_record(first.machine_id)['is_active']


def test_database_rotation_itself_requires_authorization():
    register()
    with pytest.raises(ValueError):
        db_manager.register_or_get_client('review-device-123')


def test_cached_device_revocation_and_rotation_take_effect_immediately():
    first = register()
    assert get_cached_client_credentials(first.machine_id) == (first.client_secret, True)
    with db_manager.get_session() as session:
        device = session.query(ExtensionClient).first()
        device.client_secret = 'rotated-test-credential'
        session.commit()
    assert get_cached_client_credentials(first.machine_id) == ('rotated-test-credential', True)
    with db_manager.get_session() as session:
        device = session.query(ExtensionClient).first()
        device.is_active = False
        session.commit()
    assert get_cached_client_credentials(first.machine_id) == ('rotated-test-credential', False)


def test_default_jwt_is_disabled(monkeypatch):
    monkeypatch.delenv('JWT_SECRET', raising=False)
    manager = SecurityManager()
    token = jwt.encode({'sub': 'attacker', 'type': 'admin', 'exp': time.time()+60},
                       'default-jwt-secret-change-in-production', algorithm='HS256')
    assert manager.verify_jwt_token(token) is None
    with pytest.raises(ValueError):
        manager.generate_jwt_token('user')


def test_nan_timestamp_is_rejected():
    signature = hmac.new(b'test', b'nan.body', hashlib.sha256).hexdigest()
    assert not verify_extension_signature(b'body', signature, 'nan', 'test')


def feedback(sample_id='sample-1'):
    return {'id': sample_id, 'feature_schema': 2,
            'features': extract('SyntheticRandomKey0123456789', 'API_KEY="[REDACTED]"', 'API_KEY').tolist(),
            'user_action': 'confirmed_secret', 'label': 'high'}


@pytest.mark.parametrize('change', [
    {'secret_value': 'raw-key'}, {'context': 'const API_KEY="raw-key"'}, {'label': 'false_positive'},
    {'features': [float('nan')] * 35}, {'features': [0.] * 34}, {'features': [2.] * 35},
])
def test_feedback_rejects_raw_data_and_bad_features(change):
    with pytest.raises(ValidationError):
        FeedbackRequest(samples=[{**feedback(), **change}])


def test_feedback_batch_limit():
    with pytest.raises(ValidationError):
        FeedbackRequest(samples=[feedback()] * 21)


def test_feedback_is_queued_idempotently_and_only_trains_after_admin_review():
    register()
    before = analyzer._model_revision()
    body = FeedbackRequest(samples=[feedback()])
    response = asyncio.run(extension_feedback(body, 'review-device-123'))
    assert response['status'] == 'queued' and response['accepted_sample_ids'] == ['sample-1']
    asyncio.run(extension_feedback(body, 'review-device-123'))
    assert analyzer._model_revision() == before
    with db_manager.get_session() as session:
        assert session.query(CommunityFeedback).count() == 1
        row = session.query(CommunityFeedback).first()
        identifier = row.id
    result = review_feedback(FeedbackReviewRequest(ids=[identifier], approve=True))
    assert result['model_updated'] and analyzer._model_revision() != before
    revised = analyzer._model_revision()
    result = review_feedback(FeedbackReviewRequest(ids=[identifier], approve=True))
    assert not result['model_updated'] and analyzer._model_revision() == revised


def test_feedback_daily_quota():
    register()
    for batch in range(5):
        body = FeedbackRequest(samples=[feedback(f'sample-{batch}-{i}') for i in range(20)])
        asyncio.run(extension_feedback(body, 'review-device-123'))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(extension_feedback(FeedbackRequest(samples=[feedback('over-quota')]), 'review-device-123'))
    assert exc.value.status_code == 429
    # Retries do not consume another quota slot.
    assert asyncio.run(extension_feedback(FeedbackRequest(samples=[feedback('sample-0-0')]), 'review-device-123'))['stored_samples'] == 1


def test_every_parameter_group_matches_numerical_derivative():
    model = analyzer.model
    features = np.linspace(.1, .9, 35)
    _, gradients = model.gradients(features, 'high')
    rng = np.random.default_rng(21)
    def loss():
        return -np.log(model.forward('', features)['probabilities'][0])
    for name, parameter in model.parameters().items():
        direction = rng.normal(size=parameter.shape)
        direction /= np.linalg.norm(direction)
        initial = parameter.copy()
        epsilon = 1e-5
        parameter[:] = initial + epsilon * direction
        plus = loss()
        parameter[:] = initial - epsilon * direction
        minus = loss()
        parameter[:] = initial
        numerical = (plus - minus) / (2 * epsilon)
        analytic = float(np.sum(gradients[name] * direction))
        assert np.isclose(analytic, numerical, rtol=2e-4, atol=2e-7), (name, analytic, numerical)


def test_checkpoint_restores_features_pool_optimizer_and_predictions(tmp_path):
    model = analyzer.model
    value = 'DO_NOT_PERSIST_THIS_RAW_KEY'
    features = extract(value, 'API_KEY="[REDACTED]"', 'API_KEY')
    model.add_training_sample(value, features, 'high')
    model.train_step(value, features, 'high')
    path = tmp_path / 'weights.json'
    model.save_model(str(path))
    assert value not in path.read_text()
    restored = CustomLLM(ModelConfig())
    restored.load_model(str(path))
    assert restored.training_samples == model.training_samples
    assert restored._adam_t == model._adam_t
    assert restored.forward('', features) == model.forward(value, features)
    model.train_step('', features, 'high')
    restored.train_step('', features, 'high')
    assert restored.forward('', features) == model.forward('', features)


def test_cache_revision_prevents_stale_verdict(monkeypatch):
    monkeypatch.setattr(analyzer.model, 'forward', lambda *args: {'prediction': 'high'})
    assert analyzer.calculate_enhanced_confidence('key', 'ctx', 'low') == 'high'
    monkeypatch.setattr(analyzer.model, 'forward', lambda *args: {'prediction': 'false_positive'})
    analyzer.revision = 'new-weights'
    assert analyzer.calculate_enhanced_confidence('key', 'ctx', 'low') == 'low'


def test_version_switch_is_independent_and_restores_default():
    original = analyzer.model
    assert analyzer.create_model_version('candidate')
    analyzer.model_versions['candidate'].layers[0].ffn.w1[:] += 1
    assert not np.array_equal(original.layers[0].ffn.w1, analyzer.model_versions['candidate'].layers[0].ffn.w1)
    assert analyzer.switch_model_version('candidate')
    candidate_revision = analyzer.revision
    assert analyzer.switch_model_version('default')
    assert analyzer.model is original and analyzer.revision != candidate_revision


def test_health_is_not_ready_for_untrained_model_and_low_prediction_is_respected():
    analyzer.model.is_trained = False
    assert health().status_code == 503
    analyzer.model.is_trained = True
    assert health().status_code == 200
    response = _build_response('low', 'HighEntropy0123456789abcd', 'STRIPE_API_KEY=', 'request')
    assert not response.is_likely_secret


def test_blacklist_requires_full_hash_consensus_and_admin_evidence(monkeypatch):
    with pytest.raises(ValidationError):
        HashSubmitRequest(hash='0123456789abcdef')
    value = 'sk-SyntheticReviewValue0123456789'
    hash_value = hashlib.sha256(json.dumps(['API_KEY', value], separators=(',', ':')).encode()).hexdigest()
    for i in range(3):
        asyncio.run(blacklist_add(HashSubmitRequest(hash=hash_value), f'machine-{i}'))
    assert db_manager.get_blacklist_hashes() == []
    monkeypatch.setattr(analyzer, 'calculate_enhanced_confidence', lambda *args: 'high')
    body = BlacklistReviewRequest(hash=hash_value, secret_value=value, context='API_KEY=', variable_name='API_KEY')
    assert review_blacklist(body)['status'] == 'promoted'
    assert db_manager.get_blacklist_hashes() == [hash_value]


def test_production_database_failure_never_falls_back(monkeypatch):
    from src.database import DatabaseManager
    import src.database as database_module
    monkeypatch.setenv('ENVIRONMENT', 'production')
    manager = DatabaseManager()
    manager.database_url = 'postgresql://synthetic.invalid/db'
    monkeypatch.setattr(database_module, 'create_engine', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('unavailable')))
    with pytest.raises(RuntimeError, match='Production database'):
        manager.initialize()
    assert manager.database_url.startswith('postgresql://') and manager.engine is None


def test_bootstrap_is_trained_and_passes_independent_validation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / 'src/models/bootstrap_v2.json.gz'
    manifest = json.loads((root / 'src/models/bootstrap_v2.manifest.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['sha256']
    path = tmp_path / 'bootstrap.json'
    path.write_bytes(gzip.decompress(checkpoint.read_bytes()))
    model = CustomLLM(ModelConfig())
    model.load_model(str(path))
    assert model.is_trained and model.training_samples
    from scripts.build_bootstrap import dataset, evaluate
    unseen = evaluate(model, dataset(2468, 2))
    assert unseen['precision'] >= .90 and unseen['recall'] >= .90 and unseen['binary_accuracy'] >= .90


def test_http_authentication_and_feedback_contract():
    import httpx
    from src.service import app
    from src.security import security_manager
    security_manager.load_api_keys()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.post('/train', json={})).status_code == 401
            registration = await client.post('/extension/register', json={'machine_id': 'http-test-device-123'})
            assert registration.status_code == 200
            assert (await client.post('/extension/register', json={'machine_id': 'http-test-device-123'})).status_code == 401
            raw = json.dumps({'samples': [feedback()]}, separators=(',', ':')).encode()
            ts = str(time.time())
            secret = registration.json()['client_secret']
            signature = hmac.new(secret.encode(), ts.encode()+b'.'+raw, hashlib.sha256).hexdigest()
            response = await client.post('/extension/feedback', content=raw, headers={
                'Content-Type': 'application/json', 'X-Machine-ID': 'http-test-device-123',
                'X-Extension-Timestamp': ts, 'X-Extension-Signature': signature})
            assert response.status_code == 200 and response.json()['status'] == 'queued'
            assert response.json()['model_updated'] is False
    asyncio.run(asyncio.wait_for(run(), timeout=15))


def test_feature_schema_matches_shared_fixture():
    rows = json.loads((Path(__file__).parent / 'feature-parity.json').read_text())
    for row in rows:
        assert np.allclose(extract(row['value'], row['context'], row['name']), row['features'], atol=1e-6)
