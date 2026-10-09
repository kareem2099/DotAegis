"""Request/Response Pydantic models."""
import math
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator, Field, ConfigDict, model_validator


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class UserCredentials(BaseModel):
    username: str
    password: str


class AnalyzeResponse(BaseModel):
    enhanced_confidence: str
    method: str
    is_likely_secret: bool
    category: str
    risk_level: str
    reasoning: List[str]
    error: Optional[str] = None
    request_id: Optional[str] = None


class AnalyzeRequest(BaseModel):
    secret_value: str = Field(min_length=1, max_length=10000)
    context: str = Field(max_length=5000)
    variable_name: Optional[str] = Field(default=None, max_length=128)
    features: Optional[List[float]] = None

    @field_validator('secret_value')
    def validate_secret_value(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('secret_value must be a non-empty string')
        if len(v) > 10000:
            raise ValueError('secret_value too long (max 10000 characters)')
        v = v.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        v = v.strip()
        if not v:
            raise ValueError('secret_value must not be blank')
        return v

    @field_validator('context')
    def validate_context(cls, v):
        if not isinstance(v, str):
            raise ValueError('context must be a string')
        if len(v) > 5000:
            raise ValueError('context too long (max 5000 characters)')
        v = v.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        return v.strip()


class TrainingSample(AnalyzeRequest):
    user_action: str
    label: str

    @field_validator('user_action')
    def validate_user_action(cls, v):
        allowed = ['confirmed_secret', 'ignored_warning', 'marked_false_positive']
        if v not in allowed:
            raise ValueError(f'user_action must be one of: {allowed}')
        return v

    @field_validator('label')
    def validate_label(cls, v):
        allowed = ['high', 'medium', 'low', 'false_positive']
        if v not in allowed:
            raise ValueError(f'label must be one of: {allowed}')
        return v


class FeedbackSample(BaseModel):
    """Privacy-safe, untrusted observations; never applied to weights automatically."""
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    feature_schema: Literal[2] = 2
    features: List[float] = Field(min_length=35, max_length=35)
    user_action: Literal['confirmed_secret', 'ignored_warning', 'marked_false_positive']
    label: Literal['high', 'medium', 'false_positive']

    @field_validator('features')
    def validate_features(cls, values):
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError('features must be finite values between 0 and 1')
        return values

    @model_validator(mode='after')
    def consistent_label(self):
        expected = {'confirmed_secret': 'high', 'ignored_warning': 'medium',
                    'marked_false_positive': 'false_positive'}[self.user_action]
        if self.label != expected:
            raise ValueError('label must match the user action')
        return self


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    samples: List[FeedbackSample] = Field(min_length=1, max_length=20)


class FeedbackReviewRequest(BaseModel):
    ids: List[int] = Field(min_length=1, max_length=20)
    approve: bool


class HashSubmitRequest(BaseModel):
    hash: str = Field(..., pattern=r'^[0-9a-f]{64}$')
    hash_version: Literal[2] = 2


class FPReportRequest(BaseModel):
    hash: str = Field(..., pattern=r'^[0-9a-f]{64}$')
    hash_version: Literal[2] = 2


class BlacklistReviewRequest(AnalyzeRequest):
    hash: str = Field(..., pattern=r'^[0-9a-f]{64}$')


class ExtensionRegisterRequest(BaseModel):
    machine_id: str = Field(..., min_length=10, max_length=128)
    vscode_version: Optional[str] = Field(default="", max_length=50)
    extension_version: Optional[str] = Field(default="", max_length=50)


class ExtensionRegisterResponse(BaseModel):
    status: str
    machine_id: str
    client_secret: str
    created_at: str
