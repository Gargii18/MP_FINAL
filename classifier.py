"""Module 4 - Random Forest classifier: predict the 2022 IUCN status from the 2021 status + taxonomic group.

Changes relative to the original script
---------------------------------------
* The console `input()` prompts are replaced by a function API that the Streamlit UI calls.
* Pre-processing is wrapped in a scikit-learn Pipeline (OneHotEncoder with handle_unknown='ignore'), so an unseen
  taxonomic group at prediction time is handled safely instead of silently being all-zeros.
* Targets are filtered to the same six statuses as the Markov analysis (the original filtered only the 2021 column).
* Cross-validation, balanced accuracy, macro-F1 and a "lookup-table" baseline are added, because with only two input
  features a single train/test split is a noisy and optimistic estimate.
* The other algorithms listed in the report's Section 3.5 (Decision Tree, Logistic Regression, Naive Bayes, SVM) are
  benchmarked against the Random Forest.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import RANK, RISK_TIER, STATES


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_xy(species: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = species[species["Status_2021"].isin(STATES) & species["Status_2022"].isin(STATES)]
    X = pd.DataFrame(
        {
            "Status_2021_rank": d["Status_2021"].map(RANK).astype(int).to_numpy(),
            "Taxonomy": d["Taxonomy"].astype(str).to_numpy(),
        }
    )
    y = d["Status_2022"].reset_index(drop=True)
    return X, y


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("tax", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Taxonomy"])],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _clean_feature_names(names) -> list[str]:
    out = []
    for n in names:
        n = n.replace("tax__Taxonomy_", "Group: ").replace("remainder__Status_2021_rank", "2021 status (ordinal)")
        out.append(n)
    return out


def _cv_splitter(y: pd.Series, n_splits: int = 5, seed: int = 0):
    min_class = int(y.value_counts().min())
    k = min(n_splits, min_class)
    if k >= 2:
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return KFold(n_splits=min(n_splits, max(2, len(y) // 2)), shuffle=True, random_state=seed)


def _lookup_baseline_cv(X: pd.DataFrame, y: pd.Series, splitter) -> float:
    """Accuracy of 'predict the most frequent 2022 status for this (2021 status, taxonomic group)' under CV."""
    scores = []
    for tr, te in splitter.split(X, y):
        key_tr = list(zip(X.iloc[tr]["Status_2021_rank"], X.iloc[tr]["Taxonomy"]))
        table = pd.Series(y.iloc[tr].to_numpy()).groupby(pd.Series(key_tr, dtype=object).to_numpy()).agg(
            lambda s: s.value_counts().idxmax()
        )
        by_status = pd.Series(y.iloc[tr].to_numpy()).groupby(X.iloc[tr]["Status_2021_rank"].to_numpy()).agg(
            lambda s: s.value_counts().idxmax()
        )
        overall = y.iloc[tr].value_counts().idxmax()
        preds = []
        for r, t in zip(X.iloc[te]["Status_2021_rank"], X.iloc[te]["Taxonomy"]):
            preds.append(table.get((r, t), by_status.get(r, overall)))
        scores.append(accuracy_score(y.iloc[te], preds))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_random_forest(
    species: pd.DataFrame,
    n_estimators: int = 100,
    test_size: float = 0.25,
    max_depth: int | None = None,
    seed: int = 0,
    balanced: bool = True,
) -> dict:
    cw = "balanced" if balanced else None
    X, y = build_xy(species)
    if len(y) < 20 or y.nunique() < 2:
        raise ValueError("Not enough usable records / classes to train a classifier (need >= 20 rows and >= 2 classes).")

    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, criterion="gini", max_depth=max_depth, random_state=seed, class_weight=cw
    )
    pipe = make_pipeline(rf).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    labels = [s for s in STATES if s in set(y)]
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    report = pd.DataFrame(classification_report(y_te, pred, labels=labels, zero_division=0, output_dict=True)).T

    majority = y_tr.value_counts().idxmax()
    baseline = float((y_te == majority).mean())

    # direction accuracy: did we get worsened / improved / same right?
    def direction(a_rank, b_status):
        b = b_status.map(RANK).to_numpy()
        return np.where(b > a_rank, "Worsened", np.where(b < a_rank, "Improved", "Same"))

    dir_acc = float(np.mean(direction(X_te["Status_2021_rank"].to_numpy(), y_te) == direction(X_te["Status_2021_rank"].to_numpy(), pd.Series(pred))))

    # Cross-validation of the full pipeline
    splitter = _cv_splitter(y, seed=seed)
    cv = cross_validate(
        make_pipeline(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, class_weight=cw)),
        X, y, cv=splitter, scoring=["accuracy", "balanced_accuracy", "f1_macro"],
    )
    lookup_cv = _lookup_baseline_cv(X, y, splitter)

    names = _clean_feature_names(pipe.named_steps["pre"].get_feature_names_out())
    importances = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)

    # Final model refit on ALL data for deployment-style predictions
    final = make_pipeline(
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, class_weight=cw)
    ).fit(X, y)

    return {
        "pipeline": final,
        "eval_pipeline": pipe,
        "classes": list(final.named_steps["clf"].classes_),
        "taxonomies": sorted(X["Taxonomy"].unique()),
        "balanced_weights": balanced,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_total": len(X),
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "direction_accuracy": dir_acc,
        "baseline_accuracy": baseline,
        "cv_accuracy": (float(cv["test_accuracy"].mean()), float(cv["test_accuracy"].std())),
        "cv_balanced_accuracy": float(cv["test_balanced_accuracy"].mean()),
        "cv_f1_macro": float(cv["test_f1_macro"].mean()),
        "lookup_cv_accuracy": lookup_cv,
        "cv_folds": splitter.get_n_splits(),
        "confusion_matrix": cm,
        "report": report,
        "importances": importances,
        "class_counts": y.value_counts().reindex(labels).fillna(0).astype(int),
    }


def compare_algorithms(species: pd.DataFrame, n_estimators: int = 100, seed: int = 0, balanced: bool = True) -> pd.DataFrame:
    """Cross-validated benchmark of the algorithms named in the report (Section 3.5)."""
    cw = "balanced" if balanced else None
    X, y = build_xy(species)
    splitter = _cv_splitter(y, seed=seed)
    models = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=seed, class_weight=cw),
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=cw),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "SVM (RBF)": SVC(class_weight=cw),
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in models.items():
            res = cross_validate(make_pipeline(est), X, y, cv=splitter, scoring=["accuracy", "balanced_accuracy", "f1_macro"])
            rows.append(
                {
                    "Algorithm": name,
                    "Accuracy": res["test_accuracy"].mean(),
                    "± SD": res["test_accuracy"].std(),
                    "Balanced accuracy": res["test_balanced_accuracy"].mean(),
                    "Macro F1": res["test_f1_macro"].mean(),
                }
            )
    rows.append(
        {
            "Algorithm": "Lookup table (majority per status × group)",
            "Accuracy": _lookup_baseline_cv(X, y, splitter),
            "± SD": np.nan,
            "Balanced accuracy": np.nan,
            "Macro F1": np.nan,
        }
    )
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_status(bundle: dict, taxonomy: str, status_2021: str) -> dict:
    """Predict the 2022 status distribution for one (taxonomy, 2021 status) pair."""
    if status_2021 not in RANK:
        raise ValueError(f"Invalid IUCN status '{status_2021}'. Use one of: {', '.join(STATES)}")
    X = pd.DataFrame({"Status_2021_rank": [RANK[status_2021]], "Taxonomy": [str(taxonomy)]})
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(X)[0]
    classes = list(pipe.named_steps["clf"].classes_)
    table = (
        pd.DataFrame({"IUCN status": classes, "Probability": proba})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    pred = table.iloc[0]["IUCN status"]
    r0 = RANK[status_2021]
    p_worse = float(sum(p for c, p in zip(classes, proba) if RANK[c] > r0))
    p_better = float(sum(p for c, p in zip(classes, proba) if RANK[c] < r0))
    p_same = float(sum(p for c, p in zip(classes, proba) if RANK[c] == r0))
    return {
        "prediction": pred,
        "confidence": float(table.iloc[0]["Probability"]),
        "risk_tier": RISK_TIER[pred],
        "table": table,
        "p_worse": p_worse,
        "p_better": p_better,
        "p_same": p_same,
        "p_cr_or_ex": float(sum(p for c, p in zip(classes, proba) if c in ("CR", "EX"))),
        "taxonomy_known": str(taxonomy) in set(bundle["taxonomies"]),
    }


def predict_batch(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction for a table with columns 'Taxonomy' and 'Status_2021' (case-insensitive)."""
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    c_tax = cols.get("taxonomy") or cols.get("group") or cols.get("class")
    c_st = cols.get("status_2021") or cols.get("status") or cols.get("iucn_status_2021")
    if c_tax is None or c_st is None:
        raise ValueError("Batch file needs columns 'Taxonomy' and 'Status_2021'.")
    out = df.copy()
    status = out[c_st].astype(str).str.strip().str.upper()
    bad = ~status.isin(STATES)
    X = pd.DataFrame({"Status_2021_rank": status.where(~bad, "LC").map(RANK).astype(int), "Taxonomy": out[c_tax].astype(str)})
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]
    out["Predicted_Status_2022"] = np.where(bad, "invalid status", pred)
    out["Confidence"] = np.where(bad, np.nan, proba.max(axis=1))
    out["Risk_Tier"] = [RISK_TIER.get(p, "n/a") for p in out["Predicted_Status_2022"]]
    out["Taxonomy_Seen_In_Training"] = out[c_tax].astype(str).isin(set(bundle["taxonomies"]))
    return out


def save_model(bundle: dict, path: str) -> None:
    joblib.dump({k: bundle[k] for k in ("pipeline", "classes", "taxonomies")}, path)
