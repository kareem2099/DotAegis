"""Training endpoint — /train and /reset."""
import hashlib
from fastapi import APIRouter, Depends, Request

from ..models import TrainingSample
from ..analyzer import analyzer
from ..security import get_api_key, check_rate_limit
from ..model import CustomLLM
from ..database import db_manager

router = APIRouter()


@router.post("/train", dependencies=[Depends(get_api_key), Depends(check_rate_limit)])
def train_model(request: TrainingSample, request_obj: Request):
    """Train with user feedback — server always recalculates features."""
    try:
        features_vector = analyzer.extract_features(
            request.secret_value, request.context, request.variable_name
        )
        features_list = features_vector.tolist()

        db_manager.store_training_sample(
            secret_hash=hashlib.sha256(request.secret_value.encode()).hexdigest(),
            context_hash=hashlib.sha256(request.context.encode()).hexdigest(),
            features=features_list,
            label=request.label,
            user_action=request.user_action,
            confidence=0.5,
            model_version=analyzer.active_version,
        )

        analyzer.model.add_training_sample(request.secret_value, features_list, request.label)

        if len(analyzer.model.training_samples) >= 5:
            result = analyzer.model.train(analyzer.model.training_samples[-10:], epochs=1)
            analyzer.save_model()
            return {
                "status": "trained",
                "samples_processed": result.get('samples_processed', 0),
                "average_loss": result.get('average_loss', 0.0),
                "model_updated": True,
                "features_calculated": len(features_list),
                "training_samples_stored": True,
            }
        return {
            "status": "sample_added",
            "total_samples": len(analyzer.model.training_samples),
            "message": "Sample added. Training starts at 5+ samples.",
            "features_calculated": len(features_list),
            "training_samples_stored": True,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/reset", dependencies=[Depends(get_api_key), Depends(check_rate_limit)])
def reset_model():
    """Reset model weights to random initialization."""
    try:
        analyzer.model = CustomLLM(analyzer.config)
        analyzer.cache_hits = analyzer.cache_misses = 0
        analyzer.save_model()
        return {"status": "reset", "message": "Model reset to initial state"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
