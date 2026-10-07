"""Drift detectors with a common fit/test interface.

Input-only detectors (see only features, so they are blind to pure concept drift):
  - KSChi2Detector: per-feature two-sample tests, Bonferroni-corrected.
  - PSIDetector: per-feature Population Stability Index with a fixed threshold.
  - DomainClassifierDetector: a classifier trained to tell reference rows from current rows.

Label-based detector (needs ground truth, so it only fires after labels arrive):
  - CalibrationDetector: Hosmer-Lemeshow test of the deployed model on newly labeled rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RARE_CATEGORY = "__rare__"


@dataclass
class DetectorResult:
    alarm: bool
    score: float
    details: Dict[str, float] = field(default_factory=dict)


def _collapse_rare(ref: pd.Series, cur: pd.Series, min_count: int) -> tuple[pd.Series, pd.Series]:
    counts = ref.value_counts()
    keep = set(counts[counts >= min_count].index)
    return ref.where(ref.isin(keep), RARE_CATEGORY), cur.where(cur.isin(keep), RARE_CATEGORY)


class KSChi2Detector:
    """KS test for numeric features, chi-square for categorical; alarm if any feature rejects
    at family-wise level `alpha` (Bonferroni)."""

    name = "ks_chi2"

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], alpha: float = 0.05, min_count: int = 20):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.alpha, self.min_count = alpha, min_count

    def fit(self, reference: pd.DataFrame) -> "KSChi2Detector":
        self.reference = reference[self.numeric + self.categorical].copy()
        return self

    def test(self, current: pd.DataFrame) -> DetectorResult:
        pvalues: Dict[str, float] = {}
        for col in self.numeric:
            pvalues[col] = float(stats.ks_2samp(self.reference[col], current[col]).pvalue)
        for col in self.categorical:
            ref, cur = _collapse_rare(self.reference[col], current[col], self.min_count)
            table = pd.concat([ref.value_counts(), cur.value_counts()], axis=1).fillna(0).to_numpy()
            table = table[table.sum(axis=1) > 0]
            pvalues[col] = 1.0 if len(table) < 2 else float(stats.chi2_contingency(table)[1])
        n_tests = len(pvalues)
        min_p = min(pvalues.values())
        drifted = [c for c, p in pvalues.items() if p < self.alpha / n_tests]
        return DetectorResult(
            alarm=bool(drifted),
            score=min(1.0, min_p * n_tests),
            details={"n_drifted": float(len(drifted)), **{f"p_{c}": p for c, p in pvalues.items()}},
        )


def population_stability_index(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class PSIDetector:
    """Per-feature PSI on reference-quantile bins; alarm if the largest PSI reaches `threshold`.

    The conventional 0.1 / 0.25 cut-offs ignore sample size, so the false-alarm rate of
    this detector depends on how many rows are in each window.
    """

    name = "psi"

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], threshold: float = 0.1,
                 bins: int = 10, min_count: int = 20):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.threshold, self.bins, self.min_count = threshold, bins, min_count

    def fit(self, reference: pd.DataFrame) -> "PSIDetector":
        self.reference = reference[self.numeric + self.categorical].copy()
        self.edges = {}
        for col in self.numeric:
            edges = np.unique(np.quantile(self.reference[col], np.linspace(0, 1, self.bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges[col] = edges
        return self

    def _psi(self, col: str, current: pd.DataFrame) -> float:
        if col in self.edges:
            ref_hist = np.histogram(self.reference[col], self.edges[col])[0]
            cur_hist = np.histogram(current[col], self.edges[col])[0]
        else:
            ref, cur = _collapse_rare(self.reference[col], current[col], self.min_count)
            table = pd.concat([ref.value_counts(), cur.value_counts()], axis=1).fillna(0)
            ref_hist, cur_hist = table.iloc[:, 0].to_numpy(), table.iloc[:, 1].to_numpy()
        return population_stability_index(ref_hist / ref_hist.sum(), cur_hist / cur_hist.sum())

    def test(self, current: pd.DataFrame) -> DetectorResult:
        psi = {col: self._psi(col, current) for col in self.numeric + self.categorical}
        worst = max(psi.values())
        return DetectorResult(
            alarm=worst >= self.threshold,
            score=worst,
            details={f"psi_{c}": v for c, v in psi.items()},
        )


class DomainClassifierDetector:
    """Train a classifier to separate reference rows (label 0) from current rows (label 1).

    Held-out ROC-AUC near 0.5 means the windows are indistinguishable. The score is the mean
    of the per-fold held-out AUCs; each fold's AUC has the Mann-Whitney null (mean 0.5,
    variance (n0+n1+1)/(12 n0 n1)), so the mean of k folds has about 1/k of it. Pooling
    out-of-fold scores into a single AUC instead mixes differently-scaled fold models and
    inflates the false-alarm rate (we measured 10-19% at alpha 0.05).
    With thousands of rows per window that test rejects for differences too small to
    matter, so `min_auc` adds an effect-size floor: alarm only if the shift is both
    significant and at least that separable.
    """

    name = "domain_classifier"

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], alpha: float = 0.05,
                 n_splits: int = 3, max_rows: int = 5000, seed: int = 0, min_auc: float = 0.5):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.alpha, self.n_splits, self.max_rows, self.seed = alpha, n_splits, max_rows, seed
        self.min_auc = min_auc

    def fit(self, reference: pd.DataFrame) -> "DomainClassifierDetector":
        self.reference = reference[self.numeric + self.categorical].copy()
        return self

    def _encode(self, frame: pd.DataFrame, categories: Dict[str, List[str]]) -> np.ndarray:
        parts = [frame[self.numeric].to_numpy(dtype=float)]
        for col in self.categorical:
            codes = pd.Categorical(frame[col], categories=categories[col]).codes.astype(float)
            parts.append(np.where(codes < 0, np.nan, codes)[:, None])
        return np.hstack(parts)

    def test(self, current: pd.DataFrame) -> DetectorResult:
        rng = np.random.default_rng(self.seed)
        n = min(len(current), len(self.reference), self.max_rows)
        ref = self.reference.iloc[rng.choice(len(self.reference), n, replace=False)]
        cur = current.iloc[rng.choice(len(current), n, replace=False)]
        both = pd.concat([ref, cur], axis=0)
        categories = {}
        for col in self.categorical:
            counts = both[col].value_counts()
            categories[col] = counts.index[:250].tolist()  # HistGradientBoosting caps categories at 255
        x = self._encode(both, categories)
        y = np.r_[np.zeros(n), np.ones(n)]

        fold_aucs, fold_vars = [], []
        cat_mask = [False] * len(self.numeric) + [True] * len(self.categorical)
        folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for train_idx, test_idx in folds.split(x, y):
            clf = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
                categorical_features=cat_mask, random_state=self.seed,
            )
            clf.fit(x[train_idx], y[train_idx])
            fold_aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(x[test_idx])[:, 1]))
            n1 = int(y[test_idx].sum())
            n0 = len(test_idx) - n1
            fold_vars.append((n0 + n1 + 1) / (12.0 * n0 * n1))

        auc = float(np.mean(fold_aucs))
        null_sd = float(np.sqrt(np.sum(fold_vars)) / len(fold_vars))
        p_value = float(stats.norm.sf((auc - 0.5) / null_sd))
        return DetectorResult(alarm=p_value < self.alpha and auc >= self.min_auc, score=auc,
                              details={"p_value": p_value, "n_per_side": float(n)})


class CalibrationDetector:
    """Hosmer-Lemeshow test of the deployed model's calibration on labeled rows.

    Unlike mean log loss, this does not fire just because the population got riskier: if
    P(y|x) is unchanged, a calibrated model stays calibrated under pure input drift.
    Needs labels, so in a live system it can only run once those labels arrive.
    """

    name = "calibration"

    def __init__(self, model, feature_columns: Sequence[str], target: str, alpha: float = 0.05, groups: int = 10):
        self.model, self.feature_columns, self.target = model, list(feature_columns), target
        self.alpha, self.groups = alpha, groups

    def fit(self, reference: pd.DataFrame) -> "CalibrationDetector":
        return self

    def test(self, current: pd.DataFrame) -> DetectorResult:
        p = self.model.predict_proba(current[self.feature_columns])[:, 1]
        y = current[self.target].to_numpy()
        bins = np.array_split(np.argsort(p), self.groups)
        stat = 0.0
        for idx in bins:
            expected, observed, n = p[idx].sum(), y[idx].sum(), len(idx)
            variance = expected * (1 - expected / n)
            stat += (observed - expected) ** 2 / max(variance, 1e-9)
        p_value = float(stats.chi2.sf(stat, df=self.groups))
        return DetectorResult(alarm=p_value < self.alpha, score=stat,
                              details={"p_value": p_value, "mean_residual": float(np.mean(y - p))})
