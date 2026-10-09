"""Public inference weights, never replay records or optimizer state."""
import base64
import gzip
import hashlib
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from ..analyzer import analyzer

router = APIRouter()
_cached_revision = None
_cached_release = None


@router.get('/model/release')
def model_release(request: Request):
    global _cached_revision, _cached_release
    with analyzer.lock:
        model = analyzer.model
        config = model.config
        if not model.is_trained or (config.hidden_dim, config.num_layers, config.num_heads,
                                    config.num_classes) != (64, 2, 4, 4):
            raise HTTPException(503, 'Compatible trained model unavailable')
        etag = '"' + analyzer.revision + '"'
        headers = {'ETag': etag, 'Cache-Control': 'public, max-age=300'}
        if request.headers.get('if-none-match') == etag:
            return Response(status_code=304, headers=headers)
        if _cached_revision != analyzer.revision:
            payload = {'format': 'dotenvy-local-transformer-v1', 'is_trained': True,
                       'config': {'input_mode': model.INPUT_MODE, 'num_features': 35,
                                  'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4, 'num_classes': 4},
                       'parameters': {k: v.tolist() for k, v in model.parameters().items()}}
            data = gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
            _cached_release = {'manifest': {'format': payload['format'], 'feature_schema': 2,
                               'sha256': hashlib.sha256(data).hexdigest(), 'revision': analyzer.revision,
                               'num_features': 35, 'labels': model.LABELS},
                               'weights_base64': base64.b64encode(data).decode('ascii')}
            _cached_revision = analyzer.revision
        return JSONResponse(_cached_release, headers=headers)
