"""Development threshold selection and contact-budget flags."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def select_f1_threshold(y_true, scores):
    """Return the best development threshold and F1, assigning zero to undefined F1.

    Equal best F1 values choose the lowest threshold (highest recall). Performance
    reported on these same outcomes is exploratory, not an independent estimate.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(y_true)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Supply matching one-dimensional labels and finite scores')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Supply both binary classes: 0 = stay, 1 = churn')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    best = int(f1.argmax())
    return float(thresholds[best]), float(f1[best])


def flag_top_fraction(scores, customer_ids, fraction):
    """Flag floor(fraction * customers) rows, breaking score ties by CustomerID."""
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be between zero and one')
    if not scores.index.is_unique or not scores.index.equals(customer_ids.index):
        raise ValueError('Scores and customer IDs must have matching unique row indices')
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError('Customer scores must be finite')
    if customer_ids.isna().any() or not customer_ids.is_unique:
        raise ValueError('Customer IDs must be present and unique')
    ranked = pd.DataFrame({'score': scores, 'customer_id': customer_ids}).sort_values(
        ['score', 'customer_id'], ascending=[False, True], kind='stable',
    )
    flagged = pd.Series(False, index=scores.index, name='flagged')
    flagged.loc[ranked.index[:math.floor(len(scores) * fraction)]] = True
    return flagged
