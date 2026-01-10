import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

MODEL_PATH = "artifacts/model.joblib"
TEST_PATH = "data/UNSW_NB15_testing-set.parquet"


def metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return fpr, rec, prec, f1, (tn, fp, fn, tp)


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TEST_PATH)

    for col in ["id", "attack_cat"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    y_true = df["label"].astype(int).values
    X = df.drop(columns=["label"])

    y_prob = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.0, 1.0, 101)
    target_fpr = 0.10

    print("thr\tfpr\trecall\tprec\tf1\tTN FP FN TP")
    candidates = []

    for thr in thresholds:
        fpr, rec, prec, f1, cm = metrics_at_threshold(y_true, y_prob, thr)
        tn, fp, fn, tp = cm

        if thr in [0.3, 0.5, 0.7, 0.8, 0.9]:
            print(f"{thr:.2f}\t{fpr:.3f}\t{rec:.3f}\t{prec:.3f}\t{f1:.3f}\t{tn} {fp} {fn} {tp}")

        if fpr <= target_fpr:
            candidates.append((thr, fpr, rec, prec, f1, cm))

    if not candidates:
        raise RuntimeError(f"No thresholds met target FPR <= {target_fpr:.2f}. Try increasing the target.")

    best = max(candidates, key=lambda t: (t[2], t[3], t[4]))

    thr, fpr, rec, prec, f1, cm = best
    tn, fp, fn, tp = cm

    print("\n=== Selected threshold ===")
    print(f"Target FPR <= {target_fpr:.2f}")
    print(f"threshold={thr:.2f} | FPR={fpr:.3f} | recall={rec:.3f} | precision={prec:.3f} | f1={f1:.3f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    joblib.dump({"threshold": float(thr), "target_fpr": float(target_fpr)}, "artifacts/threshold.joblib")
    print("Saved threshold to artifacts/threshold.joblib")


if __name__ == "__main__":
    main()
