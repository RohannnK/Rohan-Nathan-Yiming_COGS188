"""Logistic-regression monitor (supervised learning, regularisation, class weighting).

Pipeline: StandardScaler (fitted on the training windows only) then LogisticRegression with balanced class weights,
because one robot in four is the saboteur. C is chosen by GroupKFold cross-validation grouped by episode seed, and the
decision threshold tau on the validation split: the lowest tau whose window FPR is at most ``max_fpr``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


class LogisticMonitorModel:
    """Fitted pipeline plus tau; ``score`` returns P(saboteur) per window row."""

    def __init__(self, pipeline, tau: float = 0.5, features=FEATURES):
        self.pipeline, self.tau, self.features = pipeline, tau, list(features)

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(windows[self.features])[:, 1]

    def coefficients(self) -> pd.Series:
        """Standardised coefficients: which behaviours push a window towards 'saboteur'."""
        return pd.Series(self.pipeline[-1].coef_[0], index=self.features)


def fit_logistic(train: pd.DataFrame, Cs=(0.01, 0.1, 1.0, 10.0), folds: int = 5, seed: int = 0,
                 features=FEATURES, label: str = "is_saboteur") -> LogisticMonitorModel:
    """Pick C by episode-grouped cross-validation (ROC-AUC) on the training windows, then refit on all of them."""
    X, y, groups = train[list(features)], train[label].to_numpy(bool), train["episode_seed"]

    def pipe(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=1000,
                                                                  random_state=seed))

    folds = min(folds, groups.nunique())
    auc = {C: cross_val_score(pipe(C), X, y, groups=groups, cv=GroupKFold(folds), scoring="roc_auc").mean()
           for C in Cs}
    best = max(auc, key=auc.get)
    return LogisticMonitorModel(pipe(best).fit(X, y), features=features)


def select_tau(y_true, score, max_fpr: float = 0.05) -> float:
    """Lowest threshold whose false-positive rate on the negatives is at most max_fpr (flag when score > tau)."""
    y_true = np.asarray(y_true, bool)
    neg = np.sort(np.asarray(score)[~y_true])
    if not len(neg):
        return 0.5
    return float(neg[int(np.ceil((1 - max_fpr) * len(neg))) - 1])
