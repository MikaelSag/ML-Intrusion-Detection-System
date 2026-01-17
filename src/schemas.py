from pydantic import BaseModel
from typing import List, Dict, Any


class FlowFeatures(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: str
    severity: str
    confidence: float
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
    rule_tags: List[str]
    key_features: Dict[str, float | str]
    important_features: List[str]


class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    alerts: List[BatchAlert]