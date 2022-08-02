from typing import Optional, Iterable, Any

from sklearn import metrics as sk_metrics
import pandas as pd

from src.model_testing.datasets import ClassificationDataset


def average_precision_score(
    clf, evaluation_data: ClassificationDataset, inputs: Optional[Iterable[Any]] = None
) -> float:
    """Calculate average precision score given a classification pipeline with a `predict_proba` method."""

    y_true = evaluation_data.targets
    if inputs:
        y_pred_probs = clf.predict_proba(inputs)
    else:
        y_pred_probs = clf.predict_proba(evaluation_data.text)

    return sk_metrics.average_precision_score(y_true, y_pred_probs)


def precision_recall_table(
    clf, evaluation_data: ClassificationDataset, inputs: Optional[Iterable[Any]] = None
) -> pd.DataFrame:
    """Returns a dataframe with columns 'precision', 'recall' and 'threshold'."""

    y_true = evaluation_data.targets

    if inputs:
        y_pred_probs = clf.predict_proba(inputs)
    else:
        y_pred_probs = clf.predict_proba(evaluation_data.text)

    precision, recall, thresholds = sk_metrics.precision_recall_curve(
        y_true, y_pred_probs
    )

    return pd.DataFrame(
        [
            {
                "precision": precision[idx],
                "recall": recall[idx],
                "threshold": thresholds[idx],
            }
            for idx in range(len(thresholds))
        ]
    )


def precision_recall_curve(clf, evaluation_data: ClassificationDataset):
    """Plot a precision-recall curve."""

    y_true = evaluation_data.targets
    y_pred_probs = clf.predict_proba(evaluation_data.text)

    return sk_metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred_probs)
