"""Generate predictions and evaluate spam as the positive class."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from sms_spam.models import BaselineModel

CLASS_LABELS = ("ham", "spam")
PREDICTION_COLUMNS = ("record_id", "y_true", "y_pred", "p_spam")


def predict_records(
    model: BaselineModel,
    records: pd.DataFrame,
    *,
    threshold: float | None,
) -> pd.DataFrame:
    probabilities = model.predict_proba(records["text"])
    spam_index = list(model.classes_).index("spam")
    spam_scores = probabilities[:, spam_index]
    labels = (
        model.predict(records["text"])
        if threshold is None
        else np.where(spam_scores >= threshold, "spam", "ham")
    )
    return pd.DataFrame({
        "record_id": records["record_id"].to_numpy(copy=True),
        "y_true": records["label"].to_numpy(copy=True),
        "y_pred": labels,
        "p_spam": spam_scores,
    })


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    if not set(PREDICTION_COLUMNS).issubset(predictions.columns) or predictions.empty:
        raise ValueError("Predictions must contain record_id, y_true, y_pred and p_spam.")
    if predictions["record_id"].isna().any() or predictions["record_id"].duplicated().any():
        raise ValueError("Prediction record IDs must be present and unique.")
    if set(predictions["y_true"]) != set(CLASS_LABELS):
        raise ValueError("Reference labels must contain both ham and spam.")
    if not predictions["y_pred"].isin(CLASS_LABELS).all():
        raise ValueError("Predicted labels must be ham or spam.")

    scores = predictions["p_spam"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Spam probabilities must be finite numbers in [0, 1].")

    reference, predicted = predictions["y_true"], predictions["y_pred"]
    binary_reference = reference.eq("spam").to_numpy(dtype=int)
    precision, recall, _ = precision_recall_curve(binary_reference, scores)
    matrix = confusion_matrix(reference, predicted, labels=list(CLASS_LABELS))
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        "f1_spam": float(f1_score(reference, predicted, pos_label="spam", zero_division=0)),
        "precision_spam": float(precision_score(
            reference, predicted, pos_label="spam", zero_division=0
        )),
        "recall_spam": float(recall_score(reference, predicted, pos_label="spam", zero_division=0)),
        "macro_f1": float(f1_score(reference, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(reference, predicted)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(binary_reference, scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix.tolist(),
        "class_labels": list(CLASS_LABELS),
    }
