#!/usr/bin/env python3
"""
boiler_pipeline.py - complete pipeline for the Zhejiang coal-fired boiler data (xinan_uncompleted_data.csv).

One file, three stages (run all, or one at a time):

  audit      Audit the raw 5-second file: time structure, missing/negative/constant tags, label-like columns,
             realism checks (autocorrelation, cross-correlation, principal components), drift plot.
  baselines  1-minute aggregation, chronological split, context conditioning, MAD / Isolation Forest / PCA,
             alert burden, injected-fault evaluation, alert episodes + plots, optional label-based metrics.
  scenarios  Soft-sensor and EWMA detectors compared with PCA on benches with known injected faults, plus the
             episode set (injected + natural windows) for the RAG + LLM stage.

Usage (Colab):   !python boiler_pipeline.py xinan_uncompleted_data.csv --stage all
                 !python boiler_pipeline.py xinan_uncompleted_data.csv --stage scenarios
Outputs go to ./outputs/.  Optional: tag_dictionary.csv (columns tag,name,unit) for readable names.
"""

import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, r2_score
from sklearn.preprocessing import StandardScaler

# Keep warnings visible; suppress only a specific known warning if needed.

# ==============================================================================================
# CONFIG
# ==============================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default="xinan_uncompleted_data.csv")
parser.add_argument("--stage", default="all", choices=["audit", "baselines", "scenarios", "all"])
args, _ = parser.parse_known_args()
DATA_PATH = Path(args.data if args.data.lower().endswith(".csv") else "xinan_uncompleted_data.csv")
OUT = Path("outputs")
for sub in ("", "episodes", "episodes_llm"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

TIME_COL = "date"
RESAMPLE = "1min"                      # None keeps the native 5-second rows

# Tag roles - edit once you have the data dictionary. SENSORS=None means "every other numeric column".
CONTEXT = ["PTCA_8322A.AV_0#", "PTCA_8324.AV_0#"]            # operating conditions used to explain normal variation
INDICATORS = ["ZCLCCY.AV_0#", "YCLCCY.AV_0#", "YJJWSLL.AV_0#", "ZZQBCHLL.AV_0#"]   # plotted, not model inputs
EXCLUDE = []
SENSORS = None
TAG_DICTIONARY = Path("tag_dictionary.csv")   # optional CSV: tag,name,unit
LABEL_COL = None                               # optional 0/1 label column in the data file

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
TARGET_ALERT_RATE = 0.01               # validation-percentile threshold (99th)
PERSIST_MINUTES, PERSIST_SHARE = 5, 0.8   # persistent alert: >= 80% of steps in a 5-minute window above threshold
RIDGE_ALPHA = 1.0
EWMA_LAMBDA = 0.1                      # about 10 minutes of memory at 1-minute steps
SEED = 42

# stage "baselines": injected faults (sizes in robust SD = 1.4826 x MAD), pooled repeats
INJ_MINUTES = {"burst": 5, "offset": 30, "drift": 90}
INJ_MAGNITUDES = [2, 3, 5]
INJ_REPEATS = 5
BASE_MAIN_DETECTOR, BASE_MAIN_MODE = "pca", "residual"
TOP_EPISODE_PLOTS = 20

# stage "scenarios": benches with known events
N_BENCHES, EVENTS_PER_BENCH, MIN_GAP_MIN = 3, 8, 15
SCENARIOS = {
    "stuck":       {"minutes": 45, "mag": 0.0, "cause": "Instrument fault: transmitter output frozen (stuck sensor)"},
    "offset":      {"minutes": 60, "mag": 3.0, "cause": "Instrument fault: calibration offset"},
    "drift":       {"minutes": 60, "mag": 4.0, "cause": "Instrument fault: gradual sensor drift"},
    "noise_burst": {"minutes": 30, "mag": 3.0, "cause": "Instrument fault: excessive signal noise"},
    "group_shift": {"minutes": 60, "mag": 2.5, "cause": "Process disturbance: coherent shift of three correlated tags "
                                                        "(assign the physical cause once the tag dictionary is known)"},
}
NATURAL_TOP_N, NATURAL_WINDOW_MIN = 10, 10
SCEN_MAIN_DETECTOR = "soft"            # Pre-declared detector; do not select using test injection results.

rng = np.random.default_rng(SEED)
MODES, BASE_DETS, SCEN_DETS = ["raw", "residual"], ["mad", "iforest", "pca"], ["pca", "soft", "ss_ewma"]


# ==============================================================================================
# SHARED HELPERS
# ==============================================================================================
def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def ewma(Z, lam=EWMA_LAMBDA):
    return pd.DataFrame(Z).ewm(alpha=lam, adjust=False).mean().to_numpy()


def runs(mask):
    """(start, end) index pairs of consecutive True values."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def load():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")
    raw = pd.read_csv(DATA_PATH, parse_dates=[TIME_COL])
    if TIME_COL not in raw.columns:
        raise ValueError(f"Timestamp column '{TIME_COL}' not found. Edit TIME_COL in the configuration.")
    raw = raw.dropna(subset=[TIME_COL]).set_index(TIME_COL).sort_index()
    if raw.index.has_duplicates:
        raw = raw.groupby(level=0).median(numeric_only=True)
    num = list(raw.select_dtypes("number").columns)
    labels = None
    if LABEL_COL and LABEL_COL in raw.columns:
        labels = raw[LABEL_COL]
        num = [c for c in num if c != LABEL_COL]
    df = raw[num].resample(RESAMPLE).median().ffill(limit=3) if RESAMPLE else raw[num]
    if labels is not None and RESAMPLE:
        labels = labels.resample(RESAMPLE).max().reindex(df.index).fillna(0)
    if len(df) < 100: raise ValueError("Not enough timestamped observations after resampling.")
    step = (df.index[1] - df.index[0]).total_seconds() / 60
    D = SimpleNamespace(raw=raw, raw_num=raw[num], df=df, labels=labels, step=step)
    D.steps = lambda minutes: max(1, int(round(minutes / step)))
    D.pw = D.steps(PERSIST_MINUTES)
    D.pn = max(1, int(np.ceil(PERSIST_SHARE * D.pw)))
    D.CTX = [c for c in CONTEXT if c in df.columns]
    D.IND = [c for c in INDICATORS if c in df.columns]
    D.SENS = SENSORS or [c for c in df.columns if c not in D.CTX + D.IND + EXCLUDE]
    n = len(df)
    D.i1, D.i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    D.train, D.val, D.test = df.iloc[:D.i1], df.iloc[D.i1:D.i2], df.iloc[D.i2:]
    D.names = {}
    if TAG_DICTIONARY.exists():
        d = pd.read_csv(TAG_DICTIONARY)
        D.names = {r["tag"]: f"{r['name']} [{r.get('unit', '')}]".replace(" []", "") for _, r in d.iterrows()}
    D.label_of = lambda tag: f"{D.names[tag]} ({tag})" if tag in D.names else tag
    return D


def persistent(D, alert):
    a = pd.Series(np.asarray(alert, dtype=int))
    end_flag = (a.rolling(D.pw).sum() >= D.pn).fillna(False).to_numpy()
    expanded = pd.Series(end_flag).iloc[::-1].rolling(D.pw, min_periods=1).max().iloc[::-1].astype(bool).to_numpy()
    return end_flag, expanded


# ==============================================================================================
# MODELS
# ==============================================================================================
class Prep:
    """Raw view (standardised tags) and residual view (tag minus context-model prediction, standardised)."""

    def fit(self, S, C):
        self.s_med, self.c_med = S.median(), C.median()
        S, C = S.fillna(self.s_med), C.fillna(self.c_med)
        self.raw_scaler = StandardScaler().fit(S)
        self.c_scaler = StandardScaler().fit(C)
        self.reg = LinearRegression().fit(self.c_scaler.transform(C), S.to_numpy())
        R = S.to_numpy() - self.reg.predict(self.c_scaler.transform(C))
        self.res_scaler = StandardScaler().fit(R)
        mad = 1.4826 * np.median(np.abs(S.to_numpy() - np.median(S.to_numpy(), axis=0)), axis=0)
        self.rscale = np.where(mad > 0, mad, S.std().to_numpy())     # robust tag scale, used to size faults
        # LinearRegression predicts all sensor columns together. Calculate one R²
        # per sensor; estimator.score() on a single y column causes a 1-vs-24
        # output-shape error for this multi-output model.
        pred = self.reg.predict(self.c_scaler.transform(C))
        self.r2 = np.array([
            r2_score(S.iloc[:, j].to_numpy(), pred[:, j])
            for j in range(S.shape[1])
        ])
        return self

    def view(self, mode, S, C):
        S, C = S.fillna(self.s_med), C.fillna(self.c_med)
        if mode == "raw":
            return self.raw_scaler.transform(S)
        return self.res_scaler.transform(S.to_numpy() - self.reg.predict(self.c_scaler.transform(C)))


class Detectors:
    """MAD, Isolation Forest and PCA (Hotelling T2 / SPE) on one feature view."""

    def __init__(self, X):
        self.med = np.median(X, axis=0)
        self.mad = 1.4826 * np.median(np.abs(X - self.med), axis=0) + 1e-9
        self.iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X)
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(X)
        self.lam = self.pca.explained_variance_
        t2, spe = self._t2_spe(X)
        self.t2_ref, self.spe_ref = np.quantile(t2, 0.99), np.quantile(spe, 0.99)

    def _t2_spe(self, X):
        Z = self.pca.transform(X)
        return (Z ** 2 / self.lam).sum(axis=1), ((X - self.pca.inverse_transform(Z)) ** 2).sum(axis=1)

    def scores(self, X):
        t2, spe = self._t2_spe(X)
        return {"mad": np.abs((X - self.med) / self.mad).max(axis=1), "iforest": -self.iso.score_samples(X),
                "pca": np.maximum(t2 / self.t2_ref, spe / self.spe_ref)}


class SoftSensor:
    """Each tag predicted from all OTHER tags plus context (ridge). Residual scale from blocked out-of-fold predictions."""

    def fit(self, D, S, C):
        self.cols = list(S.columns)
        k = len(self.cols)
        A = pd.concat([S, C], axis=1)
        self.med = A.median()
        A = A.fillna(self.med)
        self.mu, self.sd = A.mean(), A.std().replace(0, 1)
        Z = ((A - self.mu) / self.sd).to_numpy()
        from sklearn.model_selection import TimeSeriesSplit
        m, gap = len(Z), D.steps(30)
        R = np.full((m, k), np.nan)
        self.models = []
        splitter = TimeSeriesSplit(n_splits=5, gap=gap)
        splits = list(splitter.split(Z))
        valid_rows = np.zeros(m, dtype=bool)
        for _, valid_idx in splits:
            valid_rows[valid_idx] = True
        for j in range(k):
            Xj, y = np.delete(Z, j, axis=1), Z[:, j]
            for train_idx, valid_idx in splits:
                mdl = Ridge(alpha=RIDGE_ALPHA).fit(Xj[train_idx], y[train_idx])
                R[valid_idx, j] = y[valid_idx] - mdl.predict(Xj[valid_idx])
            self.models.append(Ridge(alpha=RIDGE_ALPHA).fit(Xj, y))
        R_oof = R[valid_rows]
        self.rscale = 1.4826 * np.median(np.abs(R_oof - np.median(R_oof, axis=0)), axis=0) + 1e-6
        E = ewma(R_oof / self.rscale)
        self.escale = 1.4826 * np.median(np.abs(E - np.median(E, axis=0)), axis=0) + 1e-6
        self.r2_oof = 1 - np.nanvar(R, axis=0) / np.nanvar(Z[:, :k][valid_rows], axis=0)
        return self

    def zscores(self, S, C):
        A = pd.concat([S, C], axis=1).fillna(self.med)
        Z = ((A - self.mu) / self.sd).to_numpy()
        pred = np.column_stack([mdl.predict(np.delete(Z, j, axis=1)) for j, mdl in enumerate(self.models)])
        return (Z[:, :len(self.cols)] - pred) / self.rscale


def fit_baselines(D):
    B = SimpleNamespace()
    B.prep = Prep().fit(D.train[D.SENS], D.train[D.CTX])
    parts = {"train": D.train, "val": D.val, "test": D.test}
    B.X = {p: {m: B.prep.view(m, part[D.SENS], part[D.CTX]) for m in MODES} for p, part in parts.items()}
    B.det = {m: Detectors(B.X["train"][m]) for m in MODES}
    B.S_val = {m: B.det[m].scores(B.X["val"][m]) for m in MODES}
    B.S_test = {m: B.det[m].scores(B.X["test"][m]) for m in MODES}
    B.thr = {(m, d): float(np.quantile(B.S_val[m][d], 1 - TARGET_ALERT_RATE)) for m in MODES for d in BASE_DETS}
    return B


# ==============================================================================================
# STAGE 1: AUDIT
# ==============================================================================================
def stage_audit(D):
    section("AUDIT 1. FILE AND TIME STRUCTURE")
    raw = D.raw
    idx = raw.index
    print(f"File: {DATA_PATH}\nRows: {len(raw):,}   Columns: {raw.shape[1] + 1} (time + {raw.shape[1]})")
    print(f"Range: {idx.min()} -> {idx.max()}   span {idx.max() - idx.min()}")
    deltas = pd.Series(idx).diff().dropna()
    print("Most common intervals between rows:\n" + deltas.value_counts().head(5).to_string())
    mode_delta = deltas.mode().iloc[0]
    expected = int((idx.max() - idx.min()) / mode_delta) + 1
    print(f"Dominant interval {mode_delta}; rows expected for one continuous stream {expected:,}, actual {len(raw):,}")
    print(f"Gaps (> 1.5 x interval): {int((deltas > 1.5 * mode_delta).sum())} | duplicate timestamps: "
          f"{int(idx.duplicated().sum())} | exact duplicate rows: {int(raw.duplicated().sum())}")

    section("AUDIT 2. VALUE QUALITY")
    num = D.raw_num
    cat = pd.DataFrame({"missing": num.isna().sum(), "missing_pct": (num.isna().mean() * 100).round(3),
                        "negatives": (num < 0).sum(), "n_unique": num.nunique(), "min": num.min(),
                        "median": num.median(), "max": num.max(), "std": num.std()})
    cat.to_csv(OUT / "column_catalog.csv")
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(cat.round(3).to_string())
    const = cat.index[cat["n_unique"] <= 1].tolist()
    print(f"\nConstant tags: {const}")
    low_res = cat.index[cat["n_unique"] < 100].tolist()
    print(f"Coarse-resolution tags (< 100 distinct values): {low_res}")
    kw = ("fault", "alarm", "trip", "fail", "maint", "status", "event", "outage", "label", "abnormal", "outlier")
    label_like = [c for c in raw.columns if any(k in c.lower() for k in kw)]
    print("Label-like columns:", label_like if label_like else "none -> no labels in this file")

    section("AUDIT 3. REALISM CHECKS (real plant data vs independent simulated series)")
    x = num.ffill()
    x = x.loc[:, x.std() > 0].dropna()
    ac = x.apply(lambda s: s.autocorr(1))
    c = x.corr().abs().to_numpy().copy()
    np.fill_diagonal(c, np.nan)
    k95 = int((PCA().fit((x - x.mean()) / x.std()).explained_variance_ratio_.cumsum() < 0.95).sum()) + 1
    print(f"Lag-1 autocorrelation: min {ac.min():.2f}  median {ac.median():.2f}  max {ac.max():.2f}  (std {ac.std():.3f})")
    print(f"Median |correlation| between tags: {np.nanmedian(c):.3f} | pairs above 0.7: {np.nanmean(c > 0.7):.1%}")
    print(f"Principal components for 95% of variance: {k95} of {x.shape[1]}")
    if ac.std() < 0.02 and np.nanmedian(c) < 0.02 and k95 > 0.8 * x.shape[1]:
        print("WARNING: identical autocorrelation, no cross-correlation, no shared structure -> looks like independent "
              "simulated series, not plant data.")
    else:
        print("OK: autocorrelation differs by tag and tags share structure, as expected for real plant data.")

    section("AUDIT 4. LEVEL SHIFT OVER THE RECORD (hourly medians)")
    h = x.resample("1h").median()
    z = (h - h.mean()) / h.std()
    first, last = z.iloc[:6].mean(), z.iloc[-6:].mean()
    print("Largest shifts between first and last 6 hours (SD of hourly medians):")
    print((last - first).sort_values(key=np.abs, ascending=False).head(8).round(2).to_string())
    ax = z.plot(figsize=(12, 6), legend=False, lw=0.8, title="Hourly median of every tag (z-scored)")
    ax.figure.savefig(OUT / "hourly_zscores.png", dpi=110)
    plt.close(ax.figure)

    card = (f"DATASET CARD (complete the TODO lines)\nFile: {DATA_PATH.name}\nRows: {len(raw):,}  Tags: {raw.shape[1]}\n"
            f"Range: {idx.min()} to {idx.max()}   Interval: {mode_delta}\nGaps: {int((deltas > 1.5 * mode_delta).sum())}  "
            f"Duplicate timestamps: {int(idx.duplicated().sum())}\nMissing values: {int(num.isna().sum().sum())}\n"
            f"Label-like columns: {label_like if label_like else 'none'}\nConstant tags: {const}\n"
            f"Autocorrelation min/median/max: {ac.min():.2f}/{ac.median():.2f}/{ac.max():.2f}; median |corr| {np.nanmedian(c):.3f}; "
            f"PCs for 95% variance: {k95} of {x.shape[1]}\nSource / licence / citation: TODO\n"
            f"Tag dictionary (meaning, unit): TODO\nHow the abnormal fraction (8.6%) was defined: TODO\n")
    (OUT / "dataset_card.txt").write_text(card)
    print("\nSaved column_catalog.csv, hourly_zscores.png, dataset_card.txt")


# ==============================================================================================
# STAGE 2: BASELINES
# ==============================================================================================
def stage_baselines(D, B):
    section("BASELINES 1. DATA AND SPLIT")
    print(f"Rows after aggregation: {len(D.df):,} at {D.step:g}-minute steps | persistence: {D.pn} of {D.pw} steps")
    for name, part in [("train", D.train), ("val", D.val), ("test", D.test)]:
        print(f"{name:<6} {len(part):>6,} rows   {part.index.min()}  ->  {part.index.max()}")
    print(f"Monitored tags: {len(D.SENS)} | context: {D.CTX} | indicators: {D.IND}")

    section("BASELINES 2. CONTEXT CONDITIONING (in-sample R2 of the context model)")
    print(pd.Series(B.prep.r2, index=D.SENS).sort_values(ascending=False).round(3).to_string())
    for m in MODES:
        print(f"PCA components kept ({m} view): {B.det[m].pca.n_components_} of {len(D.SENS)}")

    section("BASELINES 3. ALERT BURDEN (validation and test windows)")
    rows = []
    for m in MODES:
        for d in BASE_DETS:
            r = {"view": m, "detector": d, "threshold": round(B.thr[(m, d)], 3)}
            for name, sc in [("val", B.S_val), ("test", B.S_test)]:
                alert = sc[m][d] > B.thr[(m, d)]
                _, exp = persistent(D, alert)
                r[f"{name}_alert_%"] = round(100 * alert.mean(), 2)
                r[f"{name}_episodes"] = len(runs(exp))
                r[f"{name}_alert_min"] = int(exp.sum() * D.step)
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Validation spans {len(D.val) * D.step / 60:.1f} h and test {len(D.test) * D.step / 60:.1f} h: counts are small.")
    print("Test episodes are NOT automatically false alarms: the data are described as ~8.6% abnormal.")

    section("BASELINES 4. INJECTED FAULTS (test window, robust units, pooled repeats)")
    S_te, C_te = D.test[D.SENS], D.test[D.CTX]

    def inject(kind, mag):
        L = D.steps(INJ_MINUTES[kind])
        margin = 3 * D.pw
        n_ev = int(max(1, min(6, (len(S_te) - 2 * margin) // (L * 3))))
        starts = np.linspace(margin, len(S_te) - L - margin, n_ev).astype(int)
        starts = np.clip(starts + rng.integers(-max(1, L // 2), max(1, L // 2) + 1, n_ev), margin, len(S_te) - L - margin)
        vals, events = S_te.to_numpy().copy(), []
        for s in starts:
            j = int(rng.integers(len(D.SENS)))
            shape = np.linspace(0, 1, L) if kind == "drift" else np.ones(L)
            vals[s:s + L, j] += mag * B.prep.rscale[j] * rng.choice([-1, 1]) * shape
            events.append((int(s), L))
        return pd.DataFrame(vals, columns=S_te.columns, index=S_te.index), events

    rows = []
    for kind in INJ_MINUTES:
        for mag in INJ_MAGNITUDES:
            hits = {(m, d): 0 for m in MODES for d in BASE_DETS}
            delays = {(m, d): [] for m in MODES for d in BASE_DETS}
            total = 0
            for _ in range(INJ_REPEATS):
                S_inj, events = inject(kind, mag)
                total += len(events)
                for m in MODES:
                    sc = B.det[m].scores(B.prep.view(m, S_inj, C_te))
                    for d in BASE_DETS:
                        end_flag, _ = persistent(D, sc[d] > B.thr[(m, d)])
                        for s, L in events:
                            idx = np.where(end_flag[s:s + L])[0]
                            if len(idx):
                                hits[(m, d)] += 1
                                delays[(m, d)].append(idx[0] * D.step)
            for m in MODES:
                for d in BASE_DETS:
                    rows.append({"fault": kind, "minutes": INJ_MINUTES[kind], "magnitude_robust_sd": mag, "view": m,
                                 "detector": d, "events": total, "detection_rate": hits[(m, d)] / total,
                                 "median_delay_min": float(np.median(delays[(m, d)])) if delays[(m, d)] else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "injection_results.csv", index=False)
    print("Detection rate (pooled events):")
    print(res.pivot_table(index=["fault", "magnitude_robust_sd"], columns=["view", "detector"], values="detection_rate").round(2).to_string())
    print("\nMean detection rate:")
    print(res.groupby(["view", "detector"])["detection_rate"].mean().round(2).to_string())

    section(f"BASELINES 5. ALERT EPISODES ({BASE_MAIN_MODE} view, {BASE_MAIN_DETECTOR}, test window)")
    sc_main = B.S_test[BASE_MAIN_MODE][BASE_MAIN_DETECTOR]
    _, expanded = persistent(D, sc_main > B.thr[(BASE_MAIN_MODE, BASE_MAIN_DETECTOR)])
    Z, idx = B.X["test"][BASE_MAIN_MODE], D.test.index
    eps = []
    for lo, hi in runs(expanded):
        zm = Z[lo:hi + 1].mean(axis=0)
        top = np.argsort(-np.abs(zm))[:5]
        eps.append({"alert_id": f"A{len(eps) + 1:03d}", "episode_class": "detected_natural", "start": idx[lo], "end": idx[hi],
                    "duration_min": int((hi - lo + 1) * D.step), "peak_score": round(float(sc_main[lo:hi + 1].max()), 3),
                    "top_variables": "; ".join(f"{D.label_of(D.SENS[k])} ({'high' if zm[k] > 0 else 'low'}, z={zm[k]:+.1f})" for k in top),
                    "top_tags": ",".join(D.SENS[k] for k in top[:4]),
                    **{f"ctx_{c}": round(float(D.test[c].iloc[lo:hi + 1].mean()), 3) for c in D.CTX}})
    ep_df = pd.DataFrame(eps)
    ep_df.to_csv(OUT / "alert_episodes.csv", index=False)
    print(f"{len(ep_df)} episodes -> outputs/alert_episodes.csv")
    if len(ep_df):
        print(ep_df[["alert_id", "start", "duration_min", "peak_score", "top_variables"]].head(10).to_string(index=False, max_colwidth=100))
    for _, e in ep_df.head(TOP_EPISODE_PLOTS).iterrows():
        plot_episode(D.df, e, e["top_tags"].split(","), OUT / "episodes" / f"{e['alert_id']}.png", f"{e['alert_id']}")

    if D.labels is not None:
        section("BASELINES 6. LABEL-BASED EVALUATION (test window)")
        y = D.labels.iloc[D.i2:].to_numpy().astype(int)
        print(f"Test labels: {y.sum()} abnormal of {len(y)} ({100 * y.mean():.1f}%)")
        out = []
        for m in MODES:
            for d in BASE_DETS:
                sc = B.S_test[m][d]
                p, r, _ = precision_recall_curve(y, sc)
                alert = sc > B.thr[(m, d)]
                out.append({"view": m, "detector": d, "PR_AUC": round(average_precision_score(y, sc), 3),
                            "best_F1": round(float((2 * p * r / np.clip(p + r, 1e-9, None)).max()), 3),
                            "precision_at_thr": round(float((y[alert] == 1).mean()), 3) if alert.any() else np.nan,
                            "recall_at_thr": round(float(alert[y == 1].mean()), 3) if y.sum() else np.nan})
        print(pd.DataFrame(out).to_string(index=False))
        print("Check how the labels were defined: a rule on one tag makes that tag look artificially predictive.")

    joblib.dump({"sensors": D.SENS, "context": D.CTX, "resample": RESAMPLE,
                 "thresholds": {f"{m}|{d}": v for (m, d), v in B.thr.items()},
                 "raw_scaler": B.prep.raw_scaler, "context_scaler": B.prep.c_scaler, "context_regression": B.prep.reg,
                 "residual_scaler": B.prep.res_scaler, "robust_scale": B.prep.rscale,
                 "isolation_forest": {m: B.det[m].iso for m in MODES}, "pca": {m: B.det[m].pca for m in MODES},
                 "pca_refs": {m: (B.det[m].t2_ref, B.det[m].spe_ref) for m in MODES}}, OUT / "models_baselines.joblib")
    print("\nSaved: alert_episodes.csv, injection_results.csv, episodes/*.png, models_baselines.joblib")


def plot_episode(src, e, tags, path, title, pad_min=90, truth=()):
    lo, hi = e["start"] - pd.Timedelta(minutes=pad_min), e["end"] + pd.Timedelta(minutes=pad_min)
    w = src.loc[lo:hi]
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 1.8 * len(tags)), sharex=True)
    for ax, t in zip(np.atleast_1d(axes), tags):
        ax.plot(w.index, w[t], lw=0.8)
        ax.axvspan(e["start"], e["end"], color="red", alpha=0.15)
        ax.set_ylabel(("* " if t in truth else "") + t[:16], fontsize=7)
    np.atleast_1d(axes)[0].set_title(f"{title}  {e['start']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


# ==============================================================================================
# STAGE 3: SOFT SENSORS + SCENARIO BENCHES + EPISODE SET FOR THE LLM STAGE
# ==============================================================================================
def stage_scenarios(D, B):
    section("SCENARIOS 1. SOFT SENSORS (train period only)")
    soft = SoftSensor().fit(D, D.train[D.SENS], D.train[D.CTX])
    print("Out-of-fold R2 (how well each tag is predicted from the others):")
    print(pd.Series(soft.r2_oof, index=D.SENS).sort_values(ascending=False).round(2).to_string())
    print("Tags with R2 near 0 cannot be monitored through their relationship to other tags.")

    def score_all(S, C):
        Xr = B.prep.view("residual", S, C)
        z = soft.zscores(S, C)
        e = ewma(z) / soft.escale
        return {"pca": (B.det["residual"].scores(Xr)["pca"], Xr), "soft": (np.abs(z).max(axis=1), z),
                "ss_ewma": (np.abs(e).max(axis=1), e)}

    val_sc, test_sc = score_all(D.val[D.SENS], D.val[D.CTX]), score_all(D.test[D.SENS], D.test[D.CTX])
    thr = {d: float(np.quantile(val_sc[d][0], 1 - TARGET_ALERT_RATE)) for d in SCEN_DETS}
    print("\nThresholds (99th percentile of validation scores):", {d: round(v, 3) for d, v in thr.items()})

    section("SCENARIOS 2. ALERT BURDEN ON THE UNMODIFIED WINDOWS")
    rows = []
    for d in SCEN_DETS:
        r = {"detector": d, "threshold": round(thr[d], 3)}
        for name, sc in [("val", val_sc), ("test", test_sc)]:
            alert = sc[d][0] > thr[d]
            r[f"{name}_alert_%"] = round(100 * alert.mean(), 2)
            r[f"{name}_episodes"] = len(runs(persistent(D, alert)[1]))
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    section("SCENARIOS 3. BENCHES WITH KNOWN INJECTED FAULTS (test window)")
    Ste, Cte = D.test[D.SENS], D.test[D.CTX]
    corr = D.train[D.SENS].corr().to_numpy().copy()
    np.fill_diagonal(corr, 0)

    def build_bench():
        kinds = list(SCENARIOS)
        seq = [kinds[i % len(kinds)] for i in range(EVENTS_PER_BENCH)]
        rng.shuffle(seq)
        lens = [D.steps(SCENARIOS[k]["minutes"]) for k in seq]
        margin = 3 * D.pw
        free = len(Ste) - 2 * margin - sum(lens) - EVENTS_PER_BENCH * D.steps(MIN_GAP_MIN)
        if free < 0:
            raise SystemExit("Test window too short for the requested events; reduce EVENTS_PER_BENCH.")
        gaps = rng.multinomial(free, np.ones(EVENTS_PER_BENCH + 1) / (EVENTS_PER_BENCH + 1))
        vals, events, pos = Ste.to_numpy().copy(), [], margin + gaps[0]
        for i, (kind, L) in enumerate(zip(seq, lens)):
            spec, j = SCENARIOS[kind], int(rng.integers(len(D.SENS)))
            sign, tags = rng.choice([-1, 1]), [j]
            if kind == "stuck":
                vals[pos:pos + L, j] = vals[pos, j]
            elif kind == "offset":
                vals[pos:pos + L, j] += sign * spec["mag"] * B.prep.rscale[j]
            elif kind == "drift":
                vals[pos:pos + L, j] += sign * spec["mag"] * B.prep.rscale[j] * np.linspace(0, 1, L)
            elif kind == "noise_burst":
                vals[pos:pos + L, j] += rng.normal(0, spec["mag"] * B.prep.rscale[j], L)
            elif kind == "group_shift":
                tags = [j] + list(np.argsort(-np.abs(corr[j]))[:2])
                for t in tags:
                    vals[pos:pos + L, t] += sign * np.sign(corr[j, t] if t != j else 1) * spec["mag"] * B.prep.rscale[t]
            events.append({"kind": kind, "start": pos, "L": L, "tags": [D.SENS[t] for t in tags]})
            pos += L + D.steps(MIN_GAP_MIN) + gaps[i + 1]
        return pd.DataFrame(vals, columns=Ste.columns, index=Ste.index), events

    benches, event_rows, fa_rows = [], [], []
    for b in range(N_BENCHES):
        S_inj, events = build_bench()
        benches.append((S_inj, events))
        sc_all = score_all(S_inj, Cte)
        ev_mask = np.zeros(len(S_inj), bool)
        for ev in events:
            ev_mask[max(0, ev["start"] - D.steps(10)): ev["start"] + ev["L"] + D.steps(10)] = True
        for d in SCEN_DETS:
            fa_rows.append({"bench": b, "detector": d,
                            "episodes_outside_events": len(runs(persistent(D, sc_all[d][0] > thr[d])[1] & ~ev_mask))})
        for ev in events:
            s, L = ev["start"], ev["L"]
            row = {"bench": b, "kind": ev["kind"], "start": S_inj.index[s], "minutes": int(L * D.step), "tags": ",".join(ev["tags"])}
            for d in SCEN_DETS:
                sc, Z = sc_all[d]
                hit = np.where(persistent(D, sc > thr[d])[0][s:s + L])[0]
                top3 = [D.SENS[k] for k in np.argsort(-np.abs(Z[s:s + L].mean(axis=0)))[:3]]
                row[f"det_{d}"] = bool(len(hit))
                row[f"delay_min_{d}"] = float(hit[0] * D.step) if len(hit) else np.nan
                row[f"hit3_{d}"] = bool(set(top3) & set(ev["tags"]))
            event_rows.append(row)
    ev_df = pd.DataFrame(event_rows)
    ev_df.to_csv(OUT / "scenario_bench_results.csv", index=False)
    print(f"{len(ev_df)} injected events over {N_BENCHES} benches.\n\nDetection rate by scenario:")
    tbl = ev_df.groupby("kind")[[f"det_{d}" for d in SCEN_DETS]].mean().round(2)
    tbl.loc["ALL"] = ev_df[[f"det_{d}" for d in SCEN_DETS]].mean().round(2)
    print(tbl.to_string())
    print("\nAmong detected events, share where an injected tag is in the top-3 attributed variables:")
    print(pd.DataFrame({d: [ev_df.loc[ev_df[f"det_{d}"], f"hit3_{d}"].mean()] for d in SCEN_DETS}, index=["top-3 hit"]).round(2).to_string())
    print("\nMedian detection delay (min):")
    print(pd.DataFrame({d: [np.nanmedian(ev_df[f"delay_min_{d}"])] for d in SCEN_DETS}, index=["delay"]).round(1).to_string())
    print("\nPersistent episodes outside injected events per bench (18 h each; may include real natural events):")
    print(pd.DataFrame(fa_rows).groupby("detector")["episodes_outside_events"].mean().round(1).to_string())

    main = SCEN_MAIN_DETECTOR
    print(f"\nMain detector for the episode set: {main}")

    section("SCENARIOS 4. EPISODE SET FOR THE RAG + LLM STAGE")

    def describe(Z, lo, hi, scores):
        zm = Z[lo:hi + 1].mean(axis=0)
        top = np.argsort(-np.abs(zm))[:5]
        return ("; ".join(f"{D.label_of(D.SENS[k])} ({'high' if zm[k] > 0 else 'low'}, z={zm[k]:+.1f})" for k in top),
                ",".join(D.SENS[k] for k in top[:4]), round(float(scores[lo:hi + 1].max()), 3))

    records = []
    for b, (S_inj, events) in enumerate(benches):
        sc, Z = score_all(S_inj, Cte)[main]
        end_flag, _ = persistent(D, sc > thr[main])
        for ev in events:
            s, L = ev["start"], ev["L"]
            tv, tt, pk = describe(Z, s, s + L - 1, sc)
            records.append({"kind": "injected", "episode_class": "injected_detected" if bool(end_flag[s:s + L].any()) else "injected_missed", "source": f"bench{b}", "start": S_inj.index[s], "end": S_inj.index[s + L - 1],
                            "duration_min": int(L * D.step), "detected": bool(end_flag[s:s + L].any()),
                            "true_scenario": ev["kind"], "true_cause": SCENARIOS[ev["kind"]]["cause"],
                            "true_tags": ",".join(ev["tags"]), "top_variables": tv, "top_tags": tt, "peak_score": pk,
                            **{f"ctx_{c}": round(float(Cte[c].iloc[s:s + L].mean()), 3) for c in D.CTX}, "_bench": b})
    full = pd.concat([D.val, D.test])
    sc_nat = np.r_[val_sc[main][0], test_sc[main][0]]
    Z_nat = np.vstack([val_sc[main][1], test_sc[main][1]])
    exp_nat = np.r_[persistent(D, val_sc[main][0] > thr[main])[1], persistent(D, test_sc[main][0] > thr[main])[1]]
    win = D.steps(NATURAL_WINDOW_MIN)
    smooth = pd.Series(sc_nat).rolling(win, center=True, min_periods=win).mean().to_numpy().copy()
    for _ in range(NATURAL_TOP_N):
        if np.all(np.isnan(smooth)):
            break
        c = int(np.nanargmax(smooth))
        lo, hi = max(0, c - win // 2), min(len(sc_nat) - 1, c + win // 2)
        tv, tt, pk = describe(Z_nat, lo, hi, sc_nat)
        records.append({"kind": "natural", "episode_class": "detected_natural" if bool(exp_nat[lo:hi + 1].any()) else "natural_unflagged_window", "source": "val" if c < len(D.val) else "test", "start": full.index[lo],
                        "end": full.index[hi], "duration_min": int((hi - lo + 1) * D.step), "detected": bool(exp_nat[lo:hi + 1].any()),
                        "true_scenario": "", "true_cause": "unknown (natural window)", "true_tags": "",
                        "top_variables": tv, "top_tags": tt, "peak_score": pk,
                        **{f"ctx_{cc}": round(float(full[cc].iloc[lo:hi + 1].mean()), 3) for cc in D.CTX}, "_bench": -1})
        smooth[max(0, c - D.steps(30)): c + D.steps(30) + 1] = np.nan
    epi = pd.DataFrame(records)
    epi.insert(0, "episode_id", [f"{'I' if k == 'injected' else 'N'}{i + 1:03d}" for i, k in enumerate(epi["kind"])])
    epi.drop(columns="_bench").to_csv(OUT / "episodes_for_llm.csv", index=False)
    print(f"{len(epi)} episodes -> outputs/episodes_for_llm.csv ({(epi.kind == 'injected').sum()} injected, "
          f"{(epi.kind == 'natural').sum()} natural windows)")
    print(epi[["episode_id", "kind", "start", "duration_min", "detected", "true_scenario", "peak_score"]].head(12).to_string(index=False))
    for _, e in epi.iterrows():
        src = benches[int(e["_bench"])][0] if e["kind"] == "injected" else full
        plot_episode(src, e, e["top_tags"].split(","), OUT / "episodes_llm" / f"{e['episode_id']}.png",
                     f"{e['episode_id']} {e['kind']} {e['true_scenario']}", pad_min=60, truth=str(e["true_tags"]).split(","))
    print("Plots in outputs/episodes_llm/ ('*' marks a truly injected tag). Review the natural windows by eye.")

    joblib.dump({"sensors": D.SENS, "context": D.CTX, "thresholds": thr, "main_detector": main, "ridge_models": soft.models,
                 "soft_mu": soft.mu, "soft_sd": soft.sd, "soft_median": soft.med, "soft_rscale": soft.rscale,
                 "soft_escale": soft.escale, "ewma_lambda": EWMA_LAMBDA}, OUT / "models_softsensor.joblib")
    print("\nSaved: scenario_bench_results.csv, episodes_for_llm.csv, models_softsensor.joblib")


# ==============================================================================================
# MAIN
# ==============================================================================================
if __name__ == "__main__":
    D = load()
    stages = ["audit", "baselines", "scenarios"] if args.stage == "all" else [args.stage]
    B = None
    for st in stages:
        if st == "audit":
            stage_audit(D)
        else:
            B = B or fit_baselines(D)
            (stage_baselines if st == "baselines" else stage_scenarios)(D, B)
    print("\nDone. Next: review outputs/episodes*/ plots, fill tag_dictionary.csv, then build the document corpus for RAG.")
