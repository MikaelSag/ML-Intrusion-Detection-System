from pydantic import BaseModel, Field
from typing import List, Dict, Any


class FlowFeatures(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of network flow features matching UNSW-NB15 schema"
    )


class PredictionResponse(BaseModel):
    prediction: str
    severity: str
    confidence: float
    attack_label: str
    rule_tags: List[str]
    important_features: List[str]


class BatchSummary(BaseModel):
    total_flows: int
    benign: int
    low: int
    medium: int
    high: int


class BatchAlert(BaseModel):
    index: int
    severity: str
    confidence: float
    attack_label: str
    rule_tags: List[str]
    key_features: Dict[str, float | str]
    important_features: List[str]


class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    alerts: List[BatchAlert]