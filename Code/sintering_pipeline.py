"""
sintering_pipeline.py - Corrected hot-pressing data pipeline:
  * BOM/whitespace-safe raw reader with portable relative-path fallback (recovers 231118/240319 if text)
  * Shared segmentation: cooler baseline (600-850 C), velocity-confirmed pressure onset, explicit no-crossing guard
  * Slice-first Savitzky-Golay derivative + analytical KWW conversion rate df/dt
  * Leak-free pre-run die_group feature (ROUND_20 vs FLAT_OR_ROUND10)
  * Explicit QC filtering before modeling (all runs still saved in run_table.csv with qc_flag)
  * Stage A: Curve-shape PCA + K-Means
  * Stage B: Scalar timescale regression (ln tau, ln t50, ln t90) with die_group, 1000/T, ln P, GPR & OOF permutation importance
  * Stage D: Fixed-conversion Arrhenius + power-law kinetics (f = 0.25, 0.40, 0.60, 0.75) with run-level bootstrap 95% CIs

Run:  python sintering_pipeline.py
"""
import os, warnings
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = "data"
REGISTRY = "registry_cleaned.csv" if os.path.exists("registry_cleaned.csv") else "registry.csv"
OUT = "outputs/fixed_pipeline"
R_GAS = 8.314                                         # J / (mol * K)
COMP = ["Mo_wt", "Fe_wt", "TiB_vol"]
T_GRID = [5, 10, 20, 40, 80, 160, 320, 640, 1200]     # s after pressure onset
CONV_LEVELS = [0.25, 0.40, 0.60, 0.75]                # fixed conversion levels for Stage D
MIN_N = 10
CRITICAL_FLAGS = ("no_load_crossing", "poor_curve_fit", "short_dwell", "no_densification", "early_preload_step")
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)


def resolve_raw_path(row):
    """Find raw file via direct path, data/ + path, or data/ + rel_path."""
    candidates = []
    raw_p = str(row.get("path", "")).strip()
    rel_p = str(row.get("rel_path", "")).strip() if pd.notna(row.get("rel_path", np.nan)) else ""
    if raw_p:
        candidates.append(raw_p)
        candidates.append(os.path.join(DATA_DIR, raw_p))
        candidates.append(os.path.join(DATA_DIR, os.path.basename(raw_p.replace("\\", "/"))))
    if rel_p:
        candidates.append(os.path.join(DATA_DIR, rel_p))
        candidates.append(rel_p)
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[0] if candidates else raw_p


def get_die_group_id(row):
    """Leak-free pre-run geometry group: 0 = ROUND >=18 mm pellets, 1 = FLAT bars or small-dia tall cylinders."""
    if "die_group_id" in row and pd.notna(row["die_group_id"]):
        return int(row["die_group_id"])
    spec = str(row.get("specimen", "")).strip().upper()
    diam = float(row["die_diam_mm"]) if pd.notna(row.get("die_diam_mm")) else np.nan
    return 0 if (spec == "ROUND" and np.isfinite(diam) and diam >= 18) else 1


# ====================== STEP 1: BOM-safe raw file reader ======================
def read_raw(path, sheet=None):
    """Return DataFrame[T_C, disp_mm] (1 row = 1 s) from text .XLS/.txt or binary .xls/.xlsx."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".txt", ".csv", ".tsv"):
        with open(path, "rb") as fb:
            head_bytes = fb.read(4096)
        is_binary = head_bytes.startswith(b"\xd0\xcf\x11\xe0") or head_bytes.startswith(b"PK")
        if not is_binary:
            text = head_bytes.decode("utf-8-sig", errors="ignore")
            lines = text.splitlines()
            hdr_idx = None
            for i, ln in enumerate(lines[:40]):
                low = ln.strip().lower()
                if low.startswith("details") or low.startswith("temperature"):
                    hdr_idx = i
                    break
            skip = (hdr_idx + 1) if hdr_idx is not None else 13
            sep = "," if (hdr_idx is not None and "," in lines[hdr_idx] and "\t" not in lines[hdr_idx]) else "\t"
            df_raw = pd.read_csv(
                path, sep=sep, skiprows=skip, header=None,
                engine="python", encoding="utf-8-sig", on_bad_lines="skip"
            )
            if df_raw.shape[1] >= 3 and pd.to_numeric(df_raw.iloc[:10, 0], errors="coerce").isna().all():
                t_col, x_col = 1, 2
            else:
                t_col, x_col = 0, 1
            d = pd.DataFrame({"T_C": df_raw.iloc[:, t_col], "disp_mm": df_raw.iloc[:, x_col]})
            d = d.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)
            if len(d) > 50:
                return d

    if ext == ".xlsx":
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
    else:
        rows = pd.read_excel(path, sheet_name=sheet or 0, header=None).values.tolist()

    k = next(
        i for i, r in enumerate(rows)
        if r and r[0] and (str(r[0]).strip().lower().startswith("details") or str(r[0]).strip().lower().startswith("temperature"))
    )
    if len(rows[k]) <= 3:
        warnings.warn(f"{os.path.basename(path)}: 3-column workbook - verify sample temperature column.")
    body = rows[k + 1:]
    t_idx = 1 if len(body[0]) >= 3 and (body[0][0] is None or str(body[0][0]).strip() == "") else 0
    d = pd.DataFrame({"T_C": [r[t_idx] for r in body], "disp_mm": [r[t_idx + 1] for r in body]})
    return d.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)


# ====================== STEP 2: shared segmentation + descriptors ======================
def segment_run(row):
    """
    Shared segmentation used by pipeline, phase figures, and curve model.
    Fixes:
      - Cooler baseline window (600 < Ts < 850 C) so early pre-loads near 900 C don't corrupt base.
      - Velocity-confirmed onset + explicit guard against silent argmax == 0.
      - Flags early pre-compaction steps (> 1.0 mm before 1150 C).
    """
    sheet = row.get("sheet", "")
    path = resolve_raw_path(row)
    d = read_raw(path, sheet if isinstance(sheet, str) and sheet else None)
    d = d.interpolate().bfill().ffill().reset_index(drop=True)
    T = d.T_C.values.astype(float)
    x_raw = d.disp_mm.values.astype(float)
    win = min(31, len(x_raw) if len(x_raw) % 2 == 1 else len(x_raw) - 1)
    x = savgol_filter(x_raw, max(5, win), 2)
    Ts = pd.Series(T).rolling(31, center=True, min_periods=1).median().values

    T_peak = float(Ts.max())
    end = int(np.where(Ts >= T_peak - 8.0)[0][-1])

    h0_hits = np.where(Ts > 600.0)[0]
    h0 = int(h0_hits[0]) if len(h0_hits) else 0
    base_mask = (Ts > 600.0) & (Ts < 850.0)
    if np.any(base_mask):
        base = float(np.median(x[base_mask]))
    else:
        base = float(np.median(x[:max(30, end // 4)]))

    hot_hits = np.where(Ts > 1000.0)[0]
    hot = int(hot_hits[0]) if len(hot_hits) else h0

    early_mask = (np.arange(len(x)) >= h0) & (np.arange(len(x)) < end) & (Ts < 1150.0)
    early_jump = float(np.max(x[early_mask] - base)) if np.any(early_mask) else 0.0

    vel_full = savgol_filter(x_raw, max(5, min(31, win)), 2, deriv=1)
    seg_x = x[hot:end] - base
    seg_v = vel_full[hot:end]
    crossed = (seg_x > 0.30) & (seg_v > 0.002)
    if not np.any(crossed):
        crossed = seg_x > 0.30
    flags = []
    if not np.any(crossed):
        flags.append("no_load_crossing")
        on = hot
    else:
        on = hot + int(np.argmax(crossed))
        while on > hot and vel_full[on - 1] > 0.002 and (x[on - 1] - base) > 0.05:
            on -= 1

    if early_jump > 1.0:
        flags.append("early_preload_step")

    return dict(T=T, Ts=Ts, x=x, x_raw=x_raw, h0=h0, on=on, end=end, T_peak=T_peak, base=base, seg_flags=flags)


def kww(t, S, tau, beta):
    return S * (1.0 - np.exp(-np.power(np.maximum(t, 0.0) / tau, np.clip(beta, 0.1, 3.0))))


def kww_dfdt(t, tau, beta):
    """Analytical conversion rate df/dt of normalized KWW curve f(t) = 1 - exp(-(t/tau)^beta)."""
    tt = np.maximum(t, 1e-6)
    z = np.power(tt / tau, beta)
    return (beta / tau) * np.power(tt / tau, beta - 1.0) * np.exp(-z)


def analyse_run(row):
    seg = segment_run(row)
    T, Ts, x, x_raw = seg["T"], seg["Ts"], seg["x"], seg["x_raw"]
    on, end, T_peak = seg["on"], seg["end"], seg["T_peak"]
    sl = slice(on, end + 1)
    t = np.arange(end + 1 - on, dtype=float)
    shrink = np.maximum(x[sl] - x[on], 0.0)

    # Slice sl BEFORE differentiating so pre-load baseline does not smear into early rates
    x_sl = x_raw[sl]
    dwin = min(101, len(x_sl) if len(x_sl) % 2 == 1 else len(x_sl) - 1)
    dxdt = savgol_filter(x_sl, max(5, dwin), 2, deriv=1) if len(x_sl) >= 5 else np.gradient(x_sl)

    tot = float(shrink[-1])
    conv = shrink / tot if tot > 0 else np.zeros_like(shrink)
    c = pd.DataFrame({"t": t, "T": T[sl], "shrink": shrink, "conv": conv, "dxdt": dxdt})

    at = lambda frac: float(t[np.argmax(shrink >= frac * tot)]) if tot > 0 else np.nan
    f = dict(
        run_id=row.run_id,
        die_group_id=get_die_group_id(row),
        T_load_C=float(Ts[on]),
        T_mean_C=float(T[sl].mean()),
        T_peak_C=float(T_peak),
        dwell_s=int(len(t)),
        total_shrink_mm=tot,
        t50_s=at(0.5),
        t90_s=at(0.9),
        peak_rate_um_s=float(1000.0 * dxdt[:min(600, len(dxdt))].max()),
        seg_flags="; ".join(seg["seg_flags"]),
    )
    for g in T_GRID:
        f[f"shape_{g}s"] = float(shrink[g] / tot) if (g < len(t) and tot > 0) else np.nan

    try:
        p, _ = curve_fit(
            kww, t[::5], shrink[::5],
            p0=[max(tot, 0.1), max(f["t50_s"], 5.0), 0.7],
            bounds=([0.0, 1.0, 0.1], [3.0 * max(tot, 1.0), 1e6, 3.0]),
            maxfev=25000,
        )
        r2 = float(r2_score(shrink[::5], kww(t[::5], *p)))
        f.update(S_fit=float(p[0]), tau_s=float(p[1]), beta=float(p[2]), fit_r2=r2)
    except Exception:
        f.update(S_fit=np.nan, tau_s=np.nan, beta=np.nan, fit_r2=np.nan)
    return f, c


# ====================== STEP 3: Stage D Fixed-Conversion Kinetics ======================
def build_conversion_table(runs_clean, curves):
    rows = []
    for _, r in runs_clean.iterrows():
        c = curves[r.run_id]
        if not (np.isfinite(r.tau_s) and np.isfinite(r.beta) and np.isfinite(r.P_MPa) and r.P_MPa > 0):
            continue
        for f_lev in CONV_LEVELS:
            t_f = float(r.tau_s * ((-np.log(1.0 - f_lev)) ** (1.0 / r.beta)))
            if t_f < 5.0 or t_f >= len(c) - 1:
                continue
            rate = float(kww_dfdt(t_f, r.tau_s, r.beta))
            if rate <= 0 or not np.isfinite(rate):
                continue
            T_at_f = float(np.interp(t_f, c.t.values, c["T"].values))
            rows.append(dict(
                run_id=r.run_id, label=r.label, die_group_id=int(r.die_group_id),
                conv_level=f_lev, T_K=T_at_f + 273.15, invT=1.0 / (T_at_f + 273.15),
                lnP=float(np.log(r.P_MPa)), ln_rate=float(np.log(rate)),
                **{k: float(r[k]) for k in COMP},
            ))
    return pd.DataFrame(rows)


def fit_kinetics_slice(df_iso, name, n_boot=500):
    if len(df_iso) < 8 or df_iso.run_id.nunique() < 4:
        return dict(subset=name, n_runs=df_iso.run_id.nunique() if len(df_iso) else 0, n_rows=len(df_iso), note="too few runs")
    df_iso = df_iso.copy()
    df_iso["invT_1e4"] = 1e4 * df_iso["invT"]  # O(1) scaling avoids sklearn >=1.6 LinearRegression tol=1e-6 rank truncation
    T_span = float(df_iso.T_K.max() - df_iso.T_K.min())
    X_cols = [c for c in ["invT_1e4", "lnP", "conv_level"] + COMP + ["die_group_id"] if df_iso[c].std() > 1e-12]
    ols = LinearRegression().fit(df_iso[X_cols], df_iso.ln_rate)
    coef = dict(zip(X_cols, ols.coef_))

    rng = np.random.default_rng(0)
    ids = df_iso.run_id.unique()
    Qs, ns = [], []
    for _ in range(n_boot):
        samp = pd.concat([df_iso[df_iso.run_id == i] for i in rng.choice(ids, len(ids), replace=True)])
        if any(samp[c].std() <= 1e-12 for c in X_cols):
            continue
        b = dict(zip(X_cols, LinearRegression().fit(samp[X_cols], samp.ln_rate).coef_))
        if "invT_1e4" in b:
            Qs.append(-b["invT_1e4"] * 1e4 * R_GAS / 1e3)
        if "lnP" in b:
            ns.append(b["lnP"])

    Q_val = (-coef["invT_1e4"] * 1e4 * R_GAS / 1e3) if ("invT_1e4" in coef and T_span > 50.0) else np.nan
    Q_lo = float(np.percentile(Qs, 2.5)) if (len(Qs) >= 20 and T_span > 50.0) else np.nan
    Q_hi = float(np.percentile(Qs, 97.5)) if (len(Qs) >= 20 and T_span > 50.0) else np.nan
    n_val = coef.get("lnP", np.nan)
    n_lo = float(np.percentile(ns, 2.5)) if len(ns) >= 20 else np.nan
    n_hi = float(np.percentile(ns, 97.5)) if len(ns) >= 20 else np.nan
    return dict(
        subset=name, n_runs=int(df_iso.run_id.nunique()), n_rows=len(df_iso), T_span_K=round(T_span, 1),
        Q_kJ_mol=round(Q_val, 1) if np.isfinite(Q_val) else np.nan,
        Q_ci_lo=round(Q_lo, 1) if np.isfinite(Q_lo) else np.nan,
        Q_ci_hi=round(Q_hi, 1) if np.isfinite(Q_hi) else np.nan,
        n_stress=round(n_val, 2) if np.isfinite(n_val) else np.nan,
        n_ci_lo=round(n_lo, 2) if np.isfinite(n_lo) else np.nan,
        n_ci_hi=round(n_hi, 2) if np.isfinite(n_hi) else np.nan,
        note="" if T_span > 50.0 else "T_span <= 50 K: Q suppressed",
    )


# ====================== STEP 4: Upgraded Stage B Models ======================
def make_stage_b_models():
    gpr_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.2)
    return {
        "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 25))),
        "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=3, random_state=0)),
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=2, random_state=0),
        "GradBoost": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
    }


def oof_permutation_importance(model_name, X, y, cv, n_repeats=30):
    """Out-of-fold permutation importance (avoids training-set memorization bias)."""
    rng = np.random.default_rng(0)
    base_pred = np.zeros(len(y))
    folds = list(cv.split(X))
    fitted = []
    for tr, te in folds:
        m = make_stage_b_models()[model_name]
        m.fit(X.iloc[tr], y.iloc[tr])
        base_pred[te] = m.predict(X.iloc[te])
        fitted.append((m, te))
    base_mae = mean_absolute_error(y, base_pred)
    imps = {}
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            perm_pred = np.zeros(len(y))
            for m, te in fitted:
                X_te = X.iloc[te].copy()
                X_te[col] = rng.permutation(X_te[col].values)
                perm_pred[te] = m.predict(X_te)
            deltas.append(mean_absolute_error(y, perm_pred) - base_mae)
        imps[col] = float(np.mean(deltas))
    return pd.Series(imps)


def main():
    reg = pd.read_csv(REGISTRY, dtype={"run_id": str})
    if "sheet" not in reg:
        reg["sheet"] = ""
    reg["sheet"] = reg["sheet"].fillna("")

    feats, curves = [], {}
    for _, row in reg.iterrows():
        try:
            f, c = analyse_run(row)
        except Exception as e:
            print("SKIPPED", row.run_id, "->", e)
            continue
        feats.append(f)
        curves[row.run_id] = c

    runs = reg.merge(pd.DataFrame(feats), on="run_id", suffixes=("", "_calc"))
    if "die_group_id_calc" in runs:
        runs["die_group_id"] = runs["die_group_id_calc"]
        runs.drop(columns=["die_group_id_calc"], inplace=True)

    runs["qc_flag"] = runs["seg_flags"].fillna("")
    def add_flag(mask, label):
        runs.loc[mask, "qc_flag"] = runs.loc[mask, "qc_flag"].apply(lambda s: f"{s}; {label}".strip("; "))

    add_flag((runs.T_peak_C - runs.T_load_C) > 60, "load_far_below_peak")
    add_flag(runs.fit_r2 < 0.90, "poor_curve_fit")
    add_flag(runs.dwell_s < 600, "short_dwell")
    add_flag(runs.total_shrink_mm < 0.5, "no_densification")
    add_flag(runs["shape_160s"] < 0.05, "delayed_onset_after_trigger")

    runs["qc_exclude"] = runs["qc_flag"].apply(lambda s: any(k in s for k in CRITICAL_FLAGS))
    runs.to_csv(f"{OUT}/run_table.csv", index=False)

    clean = runs[~runs["qc_exclude"]].copy().reset_index(drop=True)
    print(f"Analysed {len(runs)} runs; {len(clean)} passed critical QC filter ({runs['qc_exclude'].sum()} excluded).")
    if runs["qc_flag"].ne("").any():
        print("\nQC FLAGGED RUNS:")
        print(runs[runs["qc_flag"] != ""][["run_id", "label", "T_load_C", "T_peak_C", "total_shrink_mm", "fit_r2", "qc_flag", "qc_exclude"]].to_string(index=False))

    # ---- Stage A: PCA & K-Means on clean runs ----
    shp = [f"shape_{g}s" for g in T_GRID]
    S = clean.dropna(subset=shp)
    if len(S) >= 5:
        Xc = S[shp].values - S[shp].values.mean(axis=0)
        _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        explained = sv ** 2 / (sv ** 2).sum()
        km = KMeans(min(3, len(S)), n_init=20, random_state=0).fit(S[shp])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k, lab in enumerate(S.label.unique()):
            m = (S.label == lab).values
            ax[0].scatter(P[m, 0], P[m, 1], label=lab, s=50, color=plt.cm.tab20(k % 20))
        for (_, r), (px, py) in zip(S.iterrows(), P):
            ax[0].annotate(r.run_id, (px, py), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax[0].set_xlabel(f"PC1 ({100*explained[0]:.0f}%)")
        ax[0].set_ylabel(f"PC2 ({100*explained[1]:.0f}%)")
        ax[0].legend(fontsize=7)
        ax[0].set_title("Curve-shape space (QC-clean runs)")
        for (_, r), k in zip(S.iterrows(), km.labels_):
            ax[1].semilogx(T_GRID, r[shp].values.astype(float), color=f"C{k}", alpha=0.75)
        ax[1].set_xlabel("time after load (s)")
        ax[1].set_ylabel("normalized conversion f(t)")
        ax[1].set_title("Normalized curves by K-Means cluster")
        plt.tight_layout()
        plt.savefig(f"{OUT}/A_curve_patterns.png", dpi=130)
        plt.close()

    # ---- Stage B: Scalar regression with die_group + physics features ----
    clean["invT_mean"] = 1e3 / (clean["T_mean_C"] + 273.15)
    clean["lnP"] = np.log(clean["P_MPa"])
    FEATS = ["invT_mean", "lnP", "die_group_id"] + COMP
    all_res = []
    for tgt in ["tau_s", "t50_s", "t90_s"]:
        D = clean.dropna(subset=FEATS + [tgt]).reset_index(drop=True)
        if len(D) < MIN_N:
            continue
        y = np.log(D[tgt])
        schemes = {"random 5-fold": KFold(min(5, len(D)), shuffle=True, random_state=0), "leave-one-material-out": LeaveOneGroupOut()}
        for sname, cv in schemes.items():
            if sname.startswith("leave") and D.label.nunique() < 3:
                continue
            for mname, m in make_stage_b_models().items():
                pred = cross_val_predict(m, D[FEATS], y, cv=cv, groups=D.label if "leave" in sname else None)
                mae_ln = float(mean_absolute_error(y, pred))
                all_res.append(dict(
                    target=f"ln({tgt})", scheme=sname, model=mname,
                    R2=round(float(r2_score(y, pred)), 3),
                    MAE_ln=round(mae_ln, 3),
                    factor_err=round(float(np.exp(mae_ln)), 2),
                ))
        imp = oof_permutation_importance("RidgeCV", D[FEATS], y, KFold(min(5, len(D)), shuffle=True, random_state=0))
        print(f"\n[B] Out-of-fold permutation importance for ln({tgt}) (RidgeCV, delta MAE_ln):")
        print(imp.round(3).to_string())

    if all_res:
        res_df = pd.DataFrame(all_res)
        res_df.to_csv(f"{OUT}/B_model_scores.csv", index=False)
        print("\n[B] Upgraded validation scores:")
        print(res_df.to_string(index=False))

    # ---- Stage D: Fixed-conversion kinetics ----
    iso = build_conversion_table(clean, curves)
    if len(iso) >= MIN_N:
        slices = [
            ("D1: All clean runs", iso),
            ("D2: Ti-20TiB ROUND_20 controlled series", iso[iso.label == "Ti-20TiB"]),
            ("D3: 1350 C FLAT Ti-TiB bars", iso[(iso.die_group_id == 1) & (iso.T_K > 1590)]),
            ("D4: Non-TiB alloys", iso[iso.TiB_vol == 0]),
        ]
        k_rows = [fit_kinetics_slice(sub, name) for name, sub in slices]
        k_df = pd.DataFrame(k_rows)
        k_df.to_csv(f"{OUT}/D_kinetics_summary.csv", index=False)
        print("\n[D] Fixed-conversion kinetics (analytical KWW df/dt):")
        print(k_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for lev, g in iso.groupby("conv_level"):
            ax.scatter(1e4 * g.invT, g.ln_rate, c=g.lnP, cmap="viridis", label=f"f={lev:.2f}", s=40, alpha=0.85)
        ax.set_xlabel("10^4 / T (1/K)")
        ax.set_ylabel("ln(df/dt)  [1/s]")
        ax.set_title("Stage D: Fixed-conversion Arrhenius plot (colour = ln P)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/D1_arrhenius_conversion.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
