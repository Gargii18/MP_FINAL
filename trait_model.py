"""Trait-based extinction-risk model for mammals, trained on data/species_master_with_iucn_details.csv.

Target: the current IUCN category (LC, NT, VU, EN, CR) from the IUCN API (`iucn_category`).
Inputs: intrinsic and geographic traits only (body mass, geographic range, climate, human pressure, diet, activity,
life history, order, plus a few engineered range/mass features and the family/genus risk history).
Deliberately NOT used: the older `category` column, `yrcompiled`, and everything from the IUCN assessment itself
(`iucn_*`: threats, habitats, population trend, dates), because those are tied to the category (circular).
They are shown on the site as information.

Probabilities for labelled species are out-of-fold (each species is predicted by models that never saw its label),
averaged over several CV seeds.

What changed compared with the first version (all measured with the same cross-validation, see model_report.json):
* Random Forest + median imputation  ->  gradient boosting (HistGradientBoosting) that handles missing values natively,
  so far fewer trait gaps are filled with made-up medians.
* Engineered features: log range area, log body mass, range area per body mass, latitude/longitude span, island-like range.
* Family / genus target encoding (smoothed share of threatened species in the taxon), computed INSIDE each CV fold so
  no label leaks into its own prediction.
* Reports the threatened-vs-not AUC and PR-AUC averaged over several CV seeds, and re-measures the old Random Forest
  in the same run (`baseline_random_forest`) so the improvement is reproducible.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CLASSES = ["LC", "NT", "VU", "EN", "CR"]
THREATENED = ["VU", "EN", "CR"]
CATEGORICAL = ["order_", "ForStrat-Value"]
TAXON_COLS = ["family", "genus"]          # target-encoded inside the model, never used as raw categories
EXCLUDE = {"species_key", "yrcompiled", "MSW3_ID", "BodyMass-SpecLevel"}
MAX_MISSING = 0.5
CV_SEEDS = (0, 1, 2)                      # 3 x 5-fold CV; use (0,) for a quicker export

L1 = {"1": "Residential & commercial development", "2": "Agriculture & aquaculture", "3": "Energy production & mining",
      "4": "Transportation & service corridors", "5": "Biological resource use", "6": "Human intrusions & disturbance",
      "7": "Natural system modifications", "8": "Invasive & problematic species, genes & diseases", "9": "Pollution",
      "10": "Geological events", "11": "Climate change & severe weather", "12": "Other threats"}
L2 = {"1_1": "Housing & urban areas", "1_2": "Commercial & industrial areas", "1_3": "Tourism & recreation areas",
      "2_1": "Crop farming", "2_2": "Wood & pulp plantations", "2_3": "Livestock farming & ranching", "2_4": "Aquaculture",
      "3_1": "Oil & gas drilling", "3_2": "Mining & quarrying", "3_3": "Renewable energy",
      "4_1": "Roads & railroads", "4_2": "Utility & service lines", "4_3": "Shipping lanes", "4_4": "Flight paths",
      "5_1": "Hunting & trapping", "5_2": "Gathering plants", "5_3": "Logging & wood harvesting", "5_4": "Fishing & harvesting aquatic resources",
      "6_1": "Recreational activities", "6_2": "War, civil unrest & military exercises", "6_3": "Work & other activities",
      "7_1": "Fire & fire suppression", "7_2": "Dams & water management", "7_3": "Other ecosystem modifications",
      "8_1": "Invasive non-native species & diseases", "8_2": "Problematic native species & diseases", "8_3": "Introduced genetic material",
      "8_4": "Problematic species of unknown origin", "8_5": "Viral & prion-induced diseases", "8_6": "Diseases of unknown cause",
      "9_1": "Domestic & urban waste water", "9_2": "Industrial & military effluents", "9_3": "Agricultural & forestry effluents",
      "9_4": "Garbage & solid waste", "9_5": "Air-borne pollutants", "9_6": "Excess energy",
      "10_1": "Volcanoes", "10_2": "Earthquakes & tsunamis", "10_3": "Avalanches & landslides",
      "11_1": "Habitat shifting & alteration", "11_2": "Droughts", "11_3": "Temperature extremes", "11_4": "Storms & flooding"}
MEASURE_MAP = {
    "1": ["Plan land use to keep key habitat out of development", "Create and enforce protected areas with buffer zones"],
    "2": ["Protect remaining habitat and keep corridors between farmland and natural areas", "Promote wildlife-friendly farming and sustainable supply chains"],
    "3": ["Require impact assessments and habitat restoration for mining and energy projects", "Keep sensitive habitat off-limits to extraction"],
    "4": ["Build wildlife crossings and fencing along roads and rail", "Route new infrastructure away from key habitat"],
    "5_1": ["Enforce hunting regulations and run anti-poaching patrols", "Support alternative livelihoods and food sources for local communities"],
    "5_3": ["Promote sustainable, certified forestry", "Protect old-growth and key forest habitat"],
    "5_4": ["Set catch limits and use wildlife-safe fishing gear"],
    "5": ["Regulate and monitor the harvest of wild resources"],
    "6_1": ["Manage visitor access and reduce disturbance in sensitive areas"],
    "6_2": ["Protect wildlife and rangers in conflict areas through security and community engagement"],
    "6_3": ["Manage research, work and other human activities to reduce disturbance"],
    "6": ["Reduce human disturbance in sensitive areas"],
    "7_1": ["Use fire management suited to the habitat"], "7_2": ["Keep natural water flows and passage around dams"],
    "7": ["Restore natural ecosystem processes"],
    "8": ["Control invasive species and monitor wildlife disease"],
    "9": ["Cut pollution at its source and clean up contaminated sites"],
    "10": ["Keep several well-connected populations to buffer natural disasters"],
    "11": ["Protect climate-resilient habitat and corridors", "Support climate mitigation and adaptation planning"],
}
IMPACT = re.compile(r":\s*(\d+)")


def _json(raw):
    try:
        v = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        return []
    return v if isinstance(v, list) else []


def threats_and_measures(raw):
    """Real IUCN threats for the species (ongoing/future, not negligible), and measures matched to those threat types."""
    items = []
    for t in _json(raw):
        score, timing = str(t.get("score") or ""), str(t.get("timing") or "")
        if score.startswith("No/Negligible") or timing.startswith("Past, Unlikely"):
            continue
        parts = str(t.get("code") or "").split("_")
        key = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        m = IMPACT.search(score)
        items.append((int(m.group(1)) if m else 0, key, (t.get("description") or {}).get("en")))
    items.sort(key=lambda x: -x[0])
    names, keys = [], []
    for _, key, desc in items:
        name = L2.get(key) or L1.get(key.split("_")[0]) or desc
        if name and name not in names:
            names.append(name)
            keys.append(key)
    measures = []
    for k in keys:
        for m in MEASURE_MAP.get(k) or MEASURE_MAP.get(k.split("_")[0], []):
            if m not in measures:
                measures.append(m)
    if measures:
        measures.append("Monitor populations and fund research to guide management")
    return names[:8], measures[:7]


def habitats(raw):
    items = sorted(_json(raw), key=lambda h: h.get("majorImportance") != "Yes")
    out = []
    for h in items:
        n = (h.get("description") or {}).get("en")
        if n and n not in out:
            out.append(n)
    return out[:6]


def load(path):
    return pd.read_csv(path, encoding_errors="replace", low_memory=False)



def prepare(df):
    """Clean the table and add engineered features. Returns (table, numeric columns, categorical columns)."""
    d = df.copy()
    num = [c for c in d.select_dtypes("number").columns if c not in EXCLUDE and not c.startswith("iucn_")]
    d[num] = d[num].replace(-999, np.nan)
    for c in ("marine", "terrestria", "freshwater"):
        d[c] = d[c].astype(float)
        num.append(c)
    num = [c for c in dict.fromkeys(num) if d[c].isna().mean() <= MAX_MISSING and d[c].notna().any()]
    num_base = list(num)
    num += [c for c in add_features(d) if c not in num]
    cat = [c for c in CATEGORICAL if c in d.columns]
    for c in cat:
        d[c] = d[c].astype(str).astype("category")   # native categorical support in the boosting model
    for c in TAXON_COLS:
        d[c] = d[c].astype(str) if c in d.columns else "unknown"
    d.attrs["num_base"] = num_base                    # trait columns of the first model (for the baseline comparison)
    return d, num, cat


def add_features(d):
    """Add engineered columns to d in place; returns the names that could be built from the available columns."""
    def col(name):
        return pd.to_numeric(d[name], errors="coerce") if name in d.columns else None
    area, mass = col("26-1_GR_Area_km2"), col("5-1_AdultBodyMass_g")
    mass2, hu = col("BodyMass-Value"), col("27-2_HuPopDen_Mean_n/km2")
    maxlat, minlat, midlat = col("26-2_GR_MaxLat_dd"), col("26-3_GR_MinLat_dd"), col("26-4_GR_MidRangeLat_dd")
    maxlon, minlon = col("26-5_GR_MaxLong_dd"), col("26-6_GR_MinLong_dd")
    new = {}
    if area is not None:
        new["log_range_area"] = np.log1p(area.clip(lower=0))
        new["island_like_range"] = (area < 5000).astype(float).where(area.notna())
    if mass is not None or mass2 is not None:
        m = mass if mass is not None else mass2
        if mass is not None and mass2 is not None:
            m = mass.fillna(mass2)
        new["log_body_mass"] = np.log1p(m.clip(lower=0))
    if area is not None and "log_body_mass" in new:
        new["range_area_per_mass"] = new["log_range_area"] - new["log_body_mass"]
    if maxlat is not None and minlat is not None:
        new["lat_span"] = maxlat - minlat
    if maxlon is not None and minlon is not None:
        new["lon_span"] = maxlon - minlon
    if midlat is not None:
        new["abs_mid_latitude"] = midlat.abs()
    if hu is not None:
        new["log_human_density"] = np.log1p(hu.clip(lower=0))
    for k, v in new.items():
        d[k] = v
    return list(new)


class TraitModel(ClassifierMixin, BaseEstimator):
    """Gradient boosting on traits + in-fold family/genus target encoding. Works inside cross_val_predict, so the
    taxon encoding of a training row never sees that row's own label."""

    def __init__(self, num=(), cat=(), tax=tuple(TAXON_COLS), smoothing=10.0, random_state=0):
        self.num, self.cat, self.tax, self.smoothing, self.random_state = num, cat, tax, smoothing, random_state

    @staticmethod
    def _smooth(keys, t, prior, k):
        g = pd.DataFrame({"k": np.asarray(keys), "t": np.asarray(t)}).groupby("k")["t"].agg(["sum", "count"])
        return (g["sum"] + k * prior) / (g["count"] + k)

    def _matrix(self, X, te):
        Z = X[list(self.num) + list(self.cat)].copy()
        for c in self.tax:
            Z[c + "_threat_rate"] = te[c]
        return Z

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        t = y.isin(THREATENED).astype(float).to_numpy()
        self.classes_ = np.array(sorted(y.unique()))
        self.prior_ = float(t.mean())
        self.maps_ = {c: self._smooth(X[c], t, self.prior_, self.smoothing) for c in self.tax}
        te = {c: np.full(len(X), self.prior_) for c in self.tax}
        for a, b in StratifiedKFold(5, shuffle=True, random_state=self.random_state + 1).split(X, t):
            for c in self.tax:
                m = self._smooth(X[c].iloc[a], t[a], self.prior_, self.smoothing)
                te[c][b] = X[c].iloc[b].map(m).fillna(self.prior_).to_numpy()
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=350, max_leaf_nodes=12, min_samples_leaf=25, l2_regularization=2.0,
            categorical_features="from_dtype", random_state=self.random_state)
        self.model_.fit(self._matrix(X, te), y)
        return self

    def predict_proba(self, X):
        te = {c: X[c].map(self.maps_[c]).fillna(self.prior_).to_numpy() for c in self.tax}
        return self.model_.predict_proba(self._matrix(X, te))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _baseline_pipe(num, cat):
    """The first version of the model (Random Forest + median imputation), kept to measure the improvement."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])


PRIOR_ALPHA = 0.5  # 0 = raw probabilities (about 3 in 4 mammals come out LC); 1 = full correction for how common each category is


def _adjust(p, prior):
    """Divide each category's probability by (its share of the training data)^PRIOR_ALPHA and renormalise,
    so the many LC species do not hide the rarer, higher-risk categories."""
    a = p / np.asarray([prior[c] for c in CLASSES]) ** PRIOR_ALPHA
    return a.div(a.sum(axis=1), axis=0)


def _scores(y, pred):
    rank = {c: i for i, c in enumerate(CLASSES)}
    rec = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return {"accuracy": round(accuracy_score(y, pred), 4), "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "within_one_category": round(float(np.mean(np.abs(y.map(rank) - pred.map(rank)) <= 1)), 4),
            "share_predicted_LC": round(float((pred == "LC").mean()), 3),
            "recall": {c: round(float(r), 3) for c, r in zip(CLASSES, rec)}}


def _threat_score(p):
    return p[THREATENED].sum(axis=1)


def _oof(model, X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    raw = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return pd.DataFrame(raw, columns=sorted(CLASSES), index=X.index)[CLASSES]


def train(df):
    """Returns (table with one row per species: sci, status, p (class -> score), labelled), report dict."""
    d, num, cat = prepare(df)
    feats = num + cat + TAXON_COLS
    lab = d["iucn_category"].isin(CLASSES)
    X, y = d.loc[lab, feats], d.loc[lab, "iucn_category"]
    prior = y.value_counts(normalize=True)
    thr = y.isin(THREATENED)
    model = TraitModel(num=num, cat=cat)

    # Out-of-fold predictions, averaged over several CV seeds
    oofs, aucs, praucs = [], [], []
    for s in CV_SEEDS:
        o = _oof(TraitModel(num=num, cat=cat, random_state=s), X, y, s)
        oofs.append(o)
        aucs.append(roc_auc_score(thr, _threat_score(_adjust(o, prior))))
        praucs.append(average_precision_score(thr, _threat_score(_adjust(o, prior))))
    raw = sum(oofs) / len(oofs)
    oof = _adjust(raw, prior)

    # The first model (Random Forest), measured in the same run for an honest before/after
    nb = [c for c in d.attrs["num_base"] if c in d.columns]
    Xb = X.copy()
    for c in cat:
        Xb[c] = Xb[c].astype(str)
    base_raw = _oof(_baseline_pipe(nb, cat), Xb, y, 0)
    base_adj = _adjust(base_raw, prior)
    baseline = {"model": "Random Forest + median imputation (first version)",
                "cv_adjusted": _scores(y, base_adj.idxmax(axis=1)),
                "cv_threatened_vs_not_auc": round(roc_auc_score(thr, _threat_score(base_adj)), 4),
                "cv_threatened_vs_not_pr_auc": round(average_precision_score(thr, _threat_score(base_adj)), 4)}

    report = {
        "species_labelled": int(lab.sum()), "n_features": len(num) + len(cat) + len(TAXON_COLS), "prior_alpha": PRIOR_ALPHA,
        "model": "Gradient boosting + engineered features + in-fold family/genus target encoding",
        "cv_seeds": list(CV_SEEDS),
        "cv_adjusted": _scores(y, oof.idxmax(axis=1)), "cv_raw_probabilities": _scores(y, raw.idxmax(axis=1)),
        "cv_threatened_vs_not_auc": round(float(np.mean(aucs)), 4),
        "cv_threatened_vs_not_auc_std": round(float(np.std(aucs)), 4),
        "cv_threatened_vs_not_pr_auc": round(float(np.mean(praucs)), 4),
        "baseline_always_LC_accuracy": round(float((y == "LC").mean()), 4),
        "baseline_random_forest": baseline,
        "class_counts": y.value_counts().to_dict()}

    model.fit(X, y)
    sub = X.sample(min(1500, len(X)), random_state=0)
    imp = permutation_importance(model, sub, y.loc[sub.index], scoring="neg_log_loss", n_repeats=3, random_state=0)
    imp = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    report["top_features"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}
    probs = _adjust(pd.DataFrame(model.predict_proba(d[feats]), columns=sorted(CLASSES), index=d.index)[CLASSES], prior)
    probs.loc[oof.index] = oof  # labelled species keep their out-of-fold scores
    out = pd.DataFrame({"sci": d["sci_name"], "status": d["iucn_category"], "labelled": lab,
                        "p": [{c: round(float(r[c]), 3) for c in CLASSES} for _, r in probs.iterrows()]}, index=d.index)
    return out, report
