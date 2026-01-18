import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile

from src.schemas import FlowFeatures, PredictionResponse, BatchPredictionResponse, BatchAlert, BatchSummary
from src.rules import tag_flow
from src.severity import classify_severity

MODEL_PATH = "artifacts/model.joblib"
COLS_PATH = "artifacts/feature_columns.joblib"

app = FastAPI(
    title="ML IDS (UNSW-NB15)",
    description=(
        "Machine-learning based Intrusion Detection System for network flow logs. "
        "Supports single-flow prediction and batch analysis with severity levels, "
        "rule-based tagging, and analyst-focused summaries."
    ),
    version="1.0.0"
)

THRESH_PATH = "artifacts/threshold.joblib"

MAX_FILE_SIZE_MB = 10

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

CATEGORICAL_COLS = ["proto", "service", "state"]
VALID_CATEGORIES = {}

try:
    preprocess = model.named_steps["preprocess"]
    for name, transformer, cols in preprocess.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["onehot"]
            for col, cats in zip(cols, encoder.categories_):
                VALID_CATEGORIES[col] = set(cats)
except Exception:
    VALID_CATEGORIES = {}

def load_uploaded_file(file: UploadFile) -> pd.DataFrame:
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.2f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."
        )

    if file.filename.endswith(".csv"):
        return pd.read_csv(file.file)
    elif file.filename.endswith(".parquet"):
        return pd.read_parquet(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

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

    for col in ["proto", "service", "state"]:
        if col in data and col in VALID_CATEGORIES:
            if data[col] not in VALID_CATEGORIES[col]:
                data[col] = "unknown"

    row = {col: data.get(col, None) for col in expected_cols}
    X = pd.DataFrame([row], columns=expected_cols)

    if X.isna().any().any():
        X = X.fillna(0)

    try:
        probs = model.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    attack_prob = float(probs[1])  # P(attack)
    severity = classify_severity(attack_prob, THRESHOLD)
    prediction = "attack" if severity != "benign" else "benign"
    confidence = attack_prob

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
        "severity": severity,
        "confidence": confidence,
        "rule_tags": tags,
        "important_features": important_features,
    }


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(file: UploadFile = File(...)):
    if model is None or expected_cols is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {load_error}"
        )

    df = load_uploaded_file(file)

    # Drop label if present
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    MAX_ROWS = 1000
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows ({len(df)}). Max allowed is {MAX_ROWS}."
        )

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols]

    try:
        probs = model.predict_proba(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    attack_probs = probs[:, 1]

    summary_counts = {
        "benign": 0,
        "low": 0,
        "medium": 0,
        "high": 0
    }

    alerts = []

    for idx, prob in enumerate(attack_probs):
        severity = classify_severity(prob, THRESHOLD)
        summary_counts[severity] += 1

        if severity == "benign":
            continue

        row = df.iloc[idx].to_dict()
        rule_tags = tag_flow(row)

        try:
            preprocess = model.named_steps["preprocess"]
            clf = model.named_steps["clf"]
            feature_names = preprocess.get_feature_names_out()
            coefs = clf.coef_[0]
            top_idx = (abs(coefs)).argsort()[-8:][::-1]
            important_features = [str(feature_names[i]) for i in top_idx]
        except Exception:
            important_features = list(row.keys())[:8]

        alerts.append(
            BatchAlert(
                index=idx,
                severity=severity,
                confidence=float(prob),
                rule_tags=rule_tags,
                key_features={
                    "proto": row.get("proto"),
                    "rate": row.get("rate"),
                    "sbytes": row.get("sbytes"),
                    "dbytes": row.get("dbytes")
                },
                important_features=important_features
            )
        )

    alerts = alerts[:100]

    summary = BatchSummary(
        total_flows=len(df),
        **summary_counts
    )

    return BatchPredictionResponse(
        summary=summary,
        alerts=alerts
    )