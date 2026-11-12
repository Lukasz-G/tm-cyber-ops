"""XGBoost detector, explained with interventional TreeSHAP over a background set."""

from __future__ import annotations

import numpy as np
import shap
import xgboost as xgb


class XGBoostDetector:
    name = "xgboost"
    threshold_rule = "predicted probability > 0.5 (log-odds margin > 0)"

    def __init__(
        self,
        n_estimators: int = 5000,
        learning_rate: float = 0.02,
        max_leaves: int = 256,
        seed: int = 0,
        n_jobs: int = -1,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_leaves=max_leaves,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="logloss",
        )
        self.clf: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostDetector:
        self.clf = xgb.XGBClassifier(**self.params)
        self.clf.fit(np.asarray(X, np.int8), np.asarray(y, np.int8))
        return self

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(np.asarray(X, np.int8), output_margin=True).astype(np.float64)

    def margin_float64(self, X: np.ndarray) -> np.ndarray:
        """The same margin summed over the trees in float64 (shap's tree ensemble).

        XGBoost accumulates the tree outputs in float32; over thousands of trees that differs from the
        float64 sum by up to about 1e-5, which is the precision TreeSHAP's attributions sum to.
        """
        return shap.TreeExplainer(self.clf).model.predict(np.asarray(X, np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(np.asarray(X, np.int8)).astype(bool)

    def attribute(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        explainer = shap.TreeExplainer(
            self.clf,
            data=np.asarray(background, np.float64),
            feature_perturbation="interventional",
            model_output="raw",
        )
        return np.asarray(explainer.shap_values(np.asarray(X, np.float64), check_additivity=False))
