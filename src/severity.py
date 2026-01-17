from typing import Literal

Severity = Literal["benign", "low", "medium", "high"]


def classify_severity(probability: float, threshold: float) -> Severity:
    if probability < threshold:
        return "benign"
    elif probability < threshold + 0.10:
        return "low"
    elif probability < threshold + 0.20:
        return "medium"
    else:
        return "high"
