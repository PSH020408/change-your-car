"""P4 — three quantile regressors (q10 / q50 / q90) behind one interface.

Why quantile regression and not a single model plus a variance guess: the
interval then comes from the data itself — 80% of true deltas should fall
between q10 and q90, and evaluate.py checks that they do. A model that says
"+0.12 s, somewhere between -0.05 and +0.40" has told the user something a
point estimate cannot: that a tyre-age effect on this corner is real but
small next to lap-to-lap noise.

Backends: LightGBM (preferred), XGBoost, or scikit-learn's histogram GBM —
same params where they map, so a run is reproducible on a machine without
the first two.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.models.dataset import FeatureSpec


def available_backends() -> list[str]:
    out = []
    for name, mod in (("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("sklearn", "sklearn.ensemble")):
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            pass
    return out


def resolve_backend(requested: str = "auto") -> str:
    have = available_backends()
    if not have:
        raise ImportError("no GBM backend: install lightgbm, xgboost or scikit-learn")
    if requested == "auto":
        return have[0]
    if requested not in have:
        raise ImportError(f"backend {requested!r} not installed; available: {have}")
    return requested


def _make(alpha: float, backend: str, p: dict, spec: FeatureSpec):
    n, lr = int(p.get("n_estimators", 600)), float(p.get("learning_rate", 0.03))
    leaves, mcs = int(p.get("num_leaves", 63)), int(p.get("min_child_samples", 50))
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=n, learning_rate=lr,
                             num_leaves=leaves, min_child_samples=mcs,
                             subsample=float(p.get("subsample", 0.8)), subsample_freq=1,
                             colsample_bytree=float(p.get("colsample_bytree", 0.8)),
                             reg_lambda=float(p.get("reg_lambda", 1.0)), verbose=-1, n_jobs=-1)
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, n_estimators=n,
                            learning_rate=lr, max_leaves=leaves, max_depth=0, grow_policy="lossguide",
                            min_child_weight=mcs, subsample=float(p.get("subsample", 0.8)),
                            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
                            reg_lambda=float(p.get("reg_lambda", 1.0)), tree_method="hist",
                            enable_categorical=True, n_jobs=-1)
    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=n,
                                             learning_rate=lr, max_leaf_nodes=leaves,
                                             min_samples_leaf=mcs, l2_regularization=float(p.get("reg_lambda", 1.0)),
                                             categorical_features=spec.categorical_mask())
    raise ValueError(backend)


@dataclass
class QuantileSet:
    spec: FeatureSpec
    quantiles: list[float]
    backend: str
    params: dict
    models: dict = None

    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        # LightGBM / XGBoost take pandas categories; sklearn wants codes
        return self.spec.codes(X) if self.backend == "sklearn" else X

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuantileSet":
        self.models = {}
        Xb = self._X(X)
        for q in self.quantiles:
            m = _make(q, self.backend, self.params, self.spec)
            m.fit(Xb, y)
            self.models[q] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xb = self._X(X)
        cols = {f"q{int(round(q * 100))}": self.models[q].predict(Xb) for q in self.quantiles}
        out = pd.DataFrame(cols, index=X.index)
        return enforce_non_crossing(out)


def enforce_non_crossing(q: pd.DataFrame) -> pd.DataFrame:
    """Three separately-fitted quantiles can cross on odd rows; sort them."""
    arr = np.sort(q.to_numpy(dtype=float), axis=1)
    return pd.DataFrame(arr, columns=q.columns, index=q.index)


# ------------------------------------------------------------ ridge baseline
def make_ridge(spec: FeatureSpec, alpha: float = 1.0):
    """The linear lower bound the GBM has to beat."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), spec.numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), spec.categorical),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def ridge_frame(spec: FeatureSpec, X: pd.DataFrame) -> pd.DataFrame:
    """Categories as strings (OneHotEncoder), unknown -> 'nan'."""
    out = X.copy()
    for c in spec.categorical:
        out[c] = out[c].astype(str)
    return out
