"""Admin-only training and explicit review of untrusted community observations."""
import copy
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from ..models import TrainingSample, FeedbackReviewRequest
from ..analyzer import analyzer
from ..security import get_api_key, check_rate_limit
from ..database import db_manager, CommunityFeedback

router = APIRouter(dependencies=[Depends(get_api_key), Depends(check_rate_limit)])


def apply_samples(samples, review_ids=None):
    """Publish a durable new revision only after every sample trains successfully."""
    with analyzer.lock:
        old_model = analyzer.model
        candidate = copy.deepcopy(old_model)
        for sample in samples:
            candidate.add_training_sample('', sample['features'], sample['label'])
        result = candidate.train(samples)
        analyzer.model = candidate
        try:
            analyzer.save_model(review_ids=review_ids)
        except Exception:
            analyzer.model = old_model
            analyzer.revision = analyzer._model_revision()
            raise HTTPException(status_code=503, detail='Model persistence failed')
        if analyzer.active_version == 'default':
            analyzer._default_model = candidate
        else:
            analyzer.model_versions[analyzer.active_version] = candidate
        return result


@router.post('/train')
def train_model(request: TrainingSample):
    features = analyzer.extract_features(request.secret_value, request.context, request.variable_name).tolist()
    result = apply_samples([{'features': features, 'label': request.label}])
    db_manager.store_training_sample(
        secret_hash=hashlib.sha256(request.secret_value.encode()).hexdigest(),
        context_hash=hashlib.sha256(request.context.encode()).hexdigest(), features=features,
        label=request.label, user_action=request.user_action, confidence=0.5,
        model_version=analyzer.active_version)
    return {'status': 'trained', **result, 'features_calculated': len(features)}


@router.get('/feedback/pending')
def pending_feedback(limit: int = 20):
    with db_manager.get_session() as session:
        rows = session.query(CommunityFeedback).filter_by(status='pending').order_by(
            CommunityFeedback.id).limit(min(max(limit, 1), 100)).all()
        return {'samples': [{'id': r.id, 'features': r.features, 'label': r.label,
                             'user_action': r.user_action} for r in rows]}


@router.post('/feedback/review')
def review_feedback(request: FeedbackReviewRequest):
    with analyzer.lock:
        with db_manager.get_session() as session:
            rows = session.query(CommunityFeedback).filter(CommunityFeedback.id.in_(request.ids)).with_for_update().all()
            pending = [r for r in rows if r.status == 'pending']
            samples = [{'features': r.features, 'label': r.label} for r in pending]
            review_ids = [r.id for r in pending]
            if not request.approve:
                for row in pending:
                    row.status = 'rejected'
                session.commit()
        # Release reader locks before the snapshot transaction marks approvals.
        # The model lock serializes reviews within the required single worker.
        if request.approve and samples:
            apply_samples(samples, review_ids=review_ids)
    return {'status': 'reviewed', 'processed': len(samples), 'model_updated': request.approve and bool(samples)}


@router.post('/reset')
def reset_model():
    """Restore the tested bootstrap instead of serving random weights."""
    import gzip, tempfile, os
    from ..model import CustomLLM, ModelConfig
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bootstrap_v2.json.gz')
    with analyzer.lock:
        model = CustomLLM(ModelConfig())
        with gzip.open(path, 'rt') as data, tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(data.read())
        try:
            model.load_model(tmp.name)
        finally:
            os.unlink(tmp.name)
        original = analyzer.model
        analyzer.model = model
        try:
            analyzer.save_model()
        except Exception:
            analyzer.model = original
            analyzer.revision = analyzer._model_revision()
            raise HTTPException(status_code=503, detail='Model persistence failed')
        analyzer._default_model = model
        analyzer.active_version = 'default'
    return {'status': 'reset', 'message': 'Restored validated bootstrap', 'is_trained': model.is_trained}
