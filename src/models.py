"""Request/Response Pydantic models."""
from typing import List, Optional
from pydantic import BaseModel, validator, Field


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
    secret_value: str
    context: str
    variable_name: Optional[str] = None
    features: Optional[List[float]] = None

    @validator('secret_value')
    def validate_secret_value(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('secret_value must be a non-empty string')
        if len(v) > 10000:
            raise ValueError('secret_value too long (max 10000 characters)')
        v = v.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        return v.strip()

    @validator('context')
    def validate_context(cls, v):
        if not isinstance(v, str):
            raise ValueError('context must be a string')
        if len(v) > 5000:
            raise ValueError('context too long (max 5000 characters)')
        v = v.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        return v.strip()


class TrainingSample(BaseModel):
    secret_value: str
    context: str
    features: Optional[List[float]] = None
    variable_name: Optional[str] = None
    user_action: str
    label: str

    @validator('user_action')
    def validate_user_action(cls, v):
        allowed = ['confirmed_secret', 'ignored_warning', 'marked_false_positive']
        if v not in allowed:
            raise ValueError(f'user_action must be one of: {allowed}')
        return v

    @validator('label')
    def validate_label(cls, v):
        allowed = ['high', 'medium', 'low', 'false_positive']
        if v not in allowed:
            raise ValueError(f'label must be one of: {allowed}')
        return v


class FeedbackRequest(BaseModel):
    samples: List[TrainingSample]


class HashSubmitRequest(BaseModel):
    hash: str = Field(..., min_length=16, max_length=16)


class FPReportRequest(BaseModel):
    hash: str = Field(..., min_length=16, max_length=16)


class ExtensionRegisterRequest(BaseModel):
    machine_id: str = Field(..., min_length=10, max_length=128)
    vscode_version: Optional[str] = Field(default="", max_length=50)
    extension_version: Optional[str] = Field(default="", max_length=50)


class ExtensionRegisterResponse(BaseModel):
    status: str
    machine_id: str
    client_secret: str
    created_at: str
