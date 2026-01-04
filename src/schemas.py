from pydantic import BaseModel
from typing import List, Dict, Any


class FlowFeatures(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    rule_tags: List[str]
    important_features: List[str]