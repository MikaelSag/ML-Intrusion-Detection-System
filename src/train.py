import os
import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


TRAIN_PATH = "data/UNSW_NB15_training-set.parquet"
TEST_PATH = "data/UNSW_NB15_testing-set.parquet"
ARTIFACT_PATH = "artifacts/model.joblib"


def _infer_label_column(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    raise ValueError("Could not find a binary label column named 'label' in dataset.")


def _drop_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_candidates = [
        "id", "attack_cat"
    ]
    existing = [c for c in drop_candidates if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Put the parquet file in data/.")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Missing {TEST_PATH}. Put the parquet file in data/.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    label_col = _infer_label_column(train_df)

    train_df = _drop_useless_cols(train_df)
    test_df = _drop_useless_cols(test_df)

    y_train = train_df[label_col].astype(int)
    X_train = train_df.drop(columns=[label_col])

    y_test = test_df[label_col].astype(int)
    X_test = test_df.drop(columns=[label_col])

    cat_cols = [c for c in X_train.columns if not is_numeric_dtype(X_train[c])]
    num_cols = [c for c in X_train.columns if is_numeric_dtype(X_train[c])]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=None,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation on UNSW test set ===")
    print(classification_report(y_test, y_pred, target_names=["benign", "attack"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC: could not compute ({e})")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"\nSaved trained pipeline to: {ARTIFACT_PATH}")

    joblib.dump(list(X_train.columns), "artifacts/feature_columns.joblib")
    print("Saved raw feature column order to: artifacts/feature_columns.joblib")


if __name__ == "__main__":
    main()
