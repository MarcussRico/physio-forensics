"""Training and evaluation.

Two evaluation protocols, and the difference between them is the point of the
project.

    random          stratified k-fold over all videos. Every forgery method
                    appears in training. This is the number most papers
                    report and it is the easy case.

    leave-one-generator-out (LOGO)
                    hold out one forgery method entirely, train on the rest,
                    test on the unseen one. This measures whether a detector
                    generalises to a generator that did not exist when it was
                    built -- the only question that matters operationally.

On top of that, an ablation over feature families (quality / coherence / all)
isolates how much of the performance comes from cross-region coherence rather
than from per-region signal quality.

The classifier is deliberately small: histogram gradient boosting over a few
dozen interpretable scalars. It handles NaN natively (regions do drop out),
trains in under a second, and every decision can be traced to a named physical
quantity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .dataset import feature_names

FAMILIES = ("quality", "coherence", "all")


def build_model(kind: str = "gb", seed: int = 0):
    if kind == "gb":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.08,
            min_samples_leaf=5, l2_regularization=1.0, random_state=seed,
        )
    if kind == "logreg":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ])
    raise ValueError(f"unknown model {kind!r}; choose 'gb' or 'logreg'")


@dataclass
class EvalResult:
    protocol: str
    family: str
    model: str
    group: str
    auc: float
    accuracy: float
    n_train: int
    n_test: int
    n_positive_test: int


def _safe_auc(y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def evaluate_random(df: pd.DataFrame, family="all", model_kind="gb",
                    folds=5, seed=0) -> list[EvalResult]:
    """Stratified k-fold over all videos (the easy protocol)."""
    cols = feature_names(df, family)
    x, y = df[cols].to_numpy(dtype=float), df["label"].to_numpy(dtype=int)

    n_splits = min(folds, int(np.bincount(y).min()))
    if n_splits < 2:
        return []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(x, y):
        model = build_model(model_kind, seed)
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]

    return [EvalResult("random", family, model_kind, "all",
                       _safe_auc(y, oof), accuracy_score(y, oof > 0.5),
                       len(y), len(y), int(y.sum()))]


def evaluate_logo(df: pd.DataFrame, family="all", model_kind="gb",
                  seed=0) -> list[EvalResult]:
    """Leave-one-generator-out: the generalisation protocol that matters.

    Real videos are split alongside the generators so that the held-out fold
    always contains both classes and the AUC stays well defined.
    """
    cols = feature_names(df, family)
    fakes = df[df.label == 1]
    reals = df[df.label == 0].reset_index(drop=True)
    generators = sorted(fakes.generator.unique())
    if len(generators) < 2:
        return []

    rng = np.random.default_rng(seed)
    real_fold = rng.integers(0, len(generators), size=len(reals))

    results = []
    for i, gen in enumerate(generators):
        test_df = pd.concat([fakes[fakes.generator == gen], reals[real_fold == i]])
        train_df = pd.concat([fakes[fakes.generator != gen], reals[real_fold != i]])
        if test_df.label.nunique() < 2 or train_df.label.nunique() < 2:
            continue

        model = build_model(model_kind, seed)
        model.fit(train_df[cols].to_numpy(dtype=float),
                  train_df.label.to_numpy(dtype=int))
        scores = model.predict_proba(test_df[cols].to_numpy(dtype=float))[:, 1]
        y_test = test_df.label.to_numpy(dtype=int)

        results.append(EvalResult(
            "logo", family, model_kind, gen,
            _safe_auc(y_test, scores), accuracy_score(y_test, scores > 0.5),
            len(train_df), len(test_df), int(y_test.sum()),
        ))

    if results:
        aucs = [r.auc for r in results if np.isfinite(r.auc)]
        results.append(EvalResult(
            "logo", family, model_kind, "MEAN",
            float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean([r.accuracy for r in results])),
            0, 0, 0,
        ))
    return results


def run_full_evaluation(df: pd.DataFrame, model_kinds=("gb", "logreg"),
                        families=FAMILIES, folds=5, seed=0) -> pd.DataFrame:
    """Every protocol x family x model combination, as one tidy table."""
    df = df[df.get("error", "").fillna("") == ""] if "error" in df else df
    rows: list[EvalResult] = []
    for model_kind in model_kinds:
        for family in families:
            if not feature_names(df, family):
                continue
            rows += evaluate_random(df, family, model_kind, folds, seed)
            rows += evaluate_logo(df, family, model_kind, seed)
    return pd.DataFrame([asdict(r) for r in rows])


def permutation_importance_report(df: pd.DataFrame, family="all",
                                  model_kind="gb", seed=0, n_repeats=10) -> pd.DataFrame:
    """Which physical quantities is the model actually using?"""
    from sklearn.inspection import permutation_importance

    cols = feature_names(df, family)
    x, y = df[cols].to_numpy(dtype=float), df["label"].to_numpy(dtype=int)
    model = build_model(model_kind, seed)
    model.fit(x, y)
    imp = permutation_importance(model, x, y, n_repeats=n_repeats,
                                 random_state=seed, scoring="roc_auc")
    return (pd.DataFrame({"feature": cols,
                          "importance": imp.importances_mean,
                          "std": imp.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def train_and_save(df: pd.DataFrame, out_path: str | Path,
                   family="all", model_kind="gb", seed=0) -> dict:
    """Fit on everything and persist the model plus its column order."""
    import joblib

    cols = feature_names(df, family)
    model = build_model(model_kind, seed)
    model.fit(df[cols].to_numpy(dtype=float), df["label"].to_numpy(dtype=int))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols, "family": family,
                 "model_kind": model_kind}, out_path)

    meta = {"path": str(out_path), "n_features": len(cols),
            "n_train": len(df), "family": family, "model_kind": model_kind}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta
