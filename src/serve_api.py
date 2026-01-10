import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.schemas import FlowFeatures, PredictionResponse
from src.rules import tag_flow

MODEL_PATH = "artifacts/model.joblib"
COLS_PATH = "artifacts/feature_columns.joblib"

app = FastAPI(title="ML IDS (UNSW-NB15)")

THRESH_PATH = "artifacts/threshold.joblib"

try:
    thr_info = joblib.load(THRESH_PATH)
    THRESHOLD = float(thr_info["threshold"])
except Exception:
    THRESHOLD = 0.5

try:
    model = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(COLS_PATH)
except Exception as e:
    model = None
    expected_cols = None
    load_error = str(e)


@app.get("/health")
def health():
    if model is None or expected_cols is None:
        return {"status": "error", "detail": f"Model not loaded: {load_error}"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowFeatures):
    if model is None or expected_cols is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    data = flow.features

    row = {col: data.get(col, None) for col in expected_cols}
    X = pd.DataFrame([row], columns=expected_cols)

    probs = model.predict_proba(X)[0]
    pred_label = int(probs[1] >= THRESHOLD)
    prediction = "attack" if pred_label == 1 else "benign"
    confidence = float(max(probs))

    important_features = []
    try:
        preprocess = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]
        feature_names = preprocess.get_feature_names_out()
        coefs = clf.coef_[0]

        top_idx = (abs(coefs)).argsort()[-8:][::-1]
        important_features = [str(feature_names[i]) for i in top_idx]
    except Exception:
        important_features = list(data.keys())[:8]

    tags = tag_flow(data)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "rule_tags": tags,
        "important_features": important_features,
    }