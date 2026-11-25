"""
curve_model_fixed.py - Corrected displacement-curve forecasting model.
Fixes over curve_model.py:
  1. Eliminates target leakage: uses pre-run die_group_id (ROUND_20 vs FLAT_OR_ROUND10) instead of (total_mm > 5).
  2. Excludes critical QC-flagged runs (early pre-load, poor fit, etc.) via shared analyse_run().
  3. Decouples normalized shape f(t) from total scale S and reports BOTH shape error and absolute mm error.
  4. Benchmarks RidgeCV-KWW, Gaussian Process Regression (GPR-KWW), PCA-RidgeCV, and Direct ExtraTrees
     against same-die mean and nearest-run baselines under LORO and LOMO.
  5. Produces calibrated 5-95% PREDICTION intervals (adding out-of-fold residual noise / GPR predictive std)
     fixing the under-coverage where real runs fell outside parameter-only bootstrap bands.

Run:  python curve_model_fixed.py
"""
import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sintering_pipeline import analyse_run, get_die_group_id, kww, COMP, REGISTRY, CRITICAL_FLAGS

DT = 20
HORIZON = 1200
EXCLUDE_QC_FLAGGED = True
OUT = "outputs/curve_model_fixed"
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# die_group_id: 0 = ROUND_20 pellets, 1 = FLAT bars / ROUND_10 tall cylinder
SCENARIOS = [
    dict(name="Ti-20TiB 1250C 32MPa (ROUND_20)", T_C=1250, P_MPa=32, TiB_vol=20, Mo_wt=0, Fe_wt=0, die_group_id=0),
    dict(name="Ti-20TiB 1300C 32MPa (ROUND_20)", T_C=1300, P_MPa=32, TiB_vol=20, Mo_wt=0, Fe_wt=0, die_group_id=0),
    dict(name="Ti-5Fe 1250C 32MPa (FLAT)",       T_C=1250, P_MPa=32, TiB_vol=0,  Mo_wt=0, Fe_wt=5, die_group_id=1),
]


def build_dataset():
    reg = pd.read_csv(REGISTRY, dtype={"run_id": str})
    if "sheet" not in reg:
        reg["sheet"] = ""
    reg["sheet"] = reg["sheet"].fillna("")
    rows, curves = [], []
    for _, row in reg.iterrows():
        try:
            f, c = analyse_run(row)
        except Exception as e:
            print("SKIPPED", row.run_id, "->", e)
            continue
        flags = f.get("seg_flags", "")
        if f["fit_r2"] < 0.90:
            flags += "; poor_curve_fit"
        if (f["T_peak_C"] - f["T_load_C"]) > 60:
            flags += "; load_far_below_peak"
        if EXCLUDE_QC_FLAGGED and any(k in flags for k in CRITICAL_FLAGS):
            print(f"EXCLUDED (QC: {flags.strip('; ')}): {row.run_id} {row.label}")
            continue
        rows.append(dict(
            run_id=row.run_id, label=row.label,
            T_C=row.T_C, T_mean_C=f["T_mean_C"], P_MPa=row.P_MPa,
            **{k: row[k] for k in COMP},
            die_group_id=get_die_group_id(row),
            total_mm=f["total_shrink_mm"], S=f["S_fit"], tau=f["tau_s"], beta=f["beta"], dwell_s=f["dwell_s"],
        ))
        curves.append(c)
    D = pd.DataFrame(rows)
    horizon = int(min(HORIZON, D.dwell_s.min() - 1))
    grid = np.arange(0, horizon + 1, DT)
    Y = np.array([np.interp(grid, c.t.values, c.shrink.values) for c in curves])
    F = Y / np.maximum(Y[:, -1:], 1e-6)
    return D, Y, F, grid


def features(df, use_measured_T=False):
    T_col = df["T_mean_C"].values if (use_measured_T and "T_mean_C" in df) else df["T_C"].values
    X = pd.DataFrame({
        "invT": 1e3 / (T_col + 273.15),
        "lnP": np.log(df["P_MPa"].values),
        "die_group_id": df["die_group_id"].values,
    }, index=df.index)
    for c in COMP:
        X[c] = df[c].values
    return X


def _gpr():
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, n_restarts_optimizer=3, random_state=0))


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 25)))


def fit_predict(name, Xtr, Dtr, Ytr, Ftr, Xte, Dte, grid):
    if name in ("parametric_ridgecv", "parametric_gpr"):
        maker = _ridge if name == "parametric_ridgecv" else _gpr
        out = {}
        for key, y in [("S", np.log(Dtr.S)), ("tau", np.log(Dtr.tau)), ("beta", Dtr.beta)]:
            out[key] = maker().fit(Xtr, y).predict(Xte)
        S, tau, b = np.exp(out["S"]), np.exp(out["tau"]), np.clip(out["beta"], 0.2, 2.5)
        Yhat = np.array([kww(grid, S[i], tau[i], b[i]) for i in range(len(Xte))])
        Fhat = Yhat / np.maximum(Yhat[:, -1:], 1e-6)  # same horizon normalization as F = Y / Y[:, -1:]
        return Yhat, Fhat

    if name == "pca_ridgecv":
        mu = Ftr.mean(axis=0)
        Fc = Ftr - mu
        _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
        k_pc = min(3, Vt.shape[0])
        Ztr = Fc @ Vt[:k_pc].T
        Zhat = np.column_stack([_ridge().fit(Xtr, Ztr[:, j]).predict(Xte) for j in range(k_pc)])
        Fhat = np.clip(np.maximum.accumulate(mu + Zhat @ Vt[:k_pc], axis=1), 0.0, 1.05)
        Shat = np.exp(_ridge().fit(Xtr, np.log(Dtr.total_mm)).predict(Xte))
        return Shat[:, None] * Fhat, Fhat

    if name == "direct_extratrees":
        m = ExtraTreesRegressor(n_estimators=200, random_state=0).fit(Xtr, Ytr)
        Yhat = np.maximum.accumulate(m.predict(Xte), axis=1)
        Fhat = Yhat / np.maximum(Yhat[:, -1:], 1e-6)
        return Yhat, Fhat

    if name == "baseline: mean curve (same die)":
        Yhat, Fhat = [], []
        for g_id in Dte.die_group_id:
            mask = (Dtr.die_group_id == g_id).values
            Yhat.append(Ytr[mask].mean(axis=0) if mask.any() else Ytr.mean(axis=0))
            Fhat.append(Ftr[mask].mean(axis=0) if mask.any() else Ftr.mean(axis=0))
        return np.array(Yhat), np.array(Fhat)

    if name == "baseline: nearest run":
        Z = StandardScaler().fit(Xtr)
        a, b = Z.transform(Xtr), Z.transform(Xte)
        idx = [int(np.argmin(((a - v) ** 2).sum(axis=1))) for v in b]
        return Ytr[idx], Ftr[idx]

    raise ValueError(name)


MODELS = [
    "parametric_ridgecv",
    "parametric_gpr",
    "pca_ridgecv",
    "direct_extratrees",
    "baseline: mean curve (same die)",
    "baseline: nearest run",
]


def cross_validate(X, D, Y, F, grid, splitter):
    preds_Y = {m: np.zeros_like(Y) for m in MODELS}
    preds_F = {m: np.zeros_like(F) for m in MODELS}
    for tr, te in splitter.split(X, groups=D.label):
        for m in MODELS:
            pY, pF = fit_predict(m, X.iloc[tr], D.iloc[tr], Y[tr], F[tr], X.iloc[te], D.iloc[te], grid)
            preds_Y[m][te] = pY
            preds_F[m][te] = pF
    return preds_Y, preds_F


def score(preds_Y, preds_F, Y, F):
    rows, per_run = [], {}
    for m in MODELS:
        rel_Y = np.sqrt(((preds_Y[m] - Y) ** 2).mean(axis=1)) / np.maximum(Y.max(axis=1), 1e-6)
        rmse_F = np.sqrt(((preds_F[m] - F) ** 2).mean(axis=1))
        per_run[m] = rel_Y
        rows.append(dict(
            model=m,
            median_mm_err_pct=100.0 * np.median(rel_Y),
            mean_mm_err_pct=100.0 * rel_Y.mean(),
            median_shape_rmse_pct=100.0 * np.median(rmse_F),
            runs_within_20pct=f"{int((rel_Y <= 0.20).sum())} of {len(rel_Y)}",
        ))
    return pd.DataFrame(rows), per_run


def predict_scenarios_calibrated(model, X, D, Y, F, grid, oof_Y, n_sim=400):
    """
    Calibrated 5-95% PREDICTION intervals: combines model parameter uncertainty with
    empirical out-of-fold residual scatter (or GPR WhiteKernel noise).
    """
    S_df = pd.DataFrame(SCENARIOS)
    Xs = features(S_df)
    central, _ = fit_predict(model, X, D, Y, F, Xs, S_df.assign(label=""), grid)

    if model == "parametric_gpr":
        means, stds = {}, {}
        for key, y in [("S", np.log(D.S)), ("tau", np.log(D.tau)), ("beta", D.beta)]:
            pipe = _gpr().fit(X, y)
            mu, sd = pipe.named_steps["gaussianprocessregressor"].predict(
                pipe.named_steps["standardscaler"].transform(Xs), return_std=True
            )
            means[key], stds[key] = mu, sd
        draws = np.zeros((n_sim, len(Xs), len(grid)))
        for s in range(n_sim):
            S_s = np.exp(rng.normal(means["S"], stds["S"]))
            tau_s = np.exp(rng.normal(means["tau"], stds["tau"]))
            b_s = np.clip(rng.normal(means["beta"], stds["beta"]), 0.2, 2.5)
            for i in range(len(Xs)):
                draws[s, i] = kww(grid, S_s[i], tau_s[i], b_s[i])
    else:
        rel_res = (Y - oof_Y) / np.maximum(oof_Y[:, -1:], 1e-6)
        draws = []
        for _ in range(min(n_sim, 200)):
            idx = rng.integers(0, len(X), len(X))
            b_curve, _ = fit_predict(model, X.iloc[idx], D.iloc[idx], Y[idx], F[idx], Xs, S_df.assign(label=""), grid)
            res_idx = rng.integers(0, len(rel_res), len(Xs))
            noisy = np.maximum(b_curve + rel_res[res_idx] * b_curve[:, -1:], 0.0)
            draws.append(np.maximum.accumulate(noisy, axis=1))
        draws = np.array(draws)

    rows = []
    for i, name in enumerate(S_df.name):
        for k, t in enumerate(grid):
            rows.append(dict(
                scenario=name, time_s=int(t),
                elongation_mm=float(central[i, k]),
                low_mm=float(np.percentile(draws[:, i, k], 5)),
                high_mm=float(np.percentile(draws[:, i, k], 95)),
            ))
    return pd.DataFrame(rows)


def main():
    D, Y, F, grid = build_dataset()
    X = features(D)
    print(f"Dataset: {len(D)} clean runs, {len(grid)} time steps (0 to {grid[-1]} s)")
    print("Pre-run die_group_id counts (0=ROUND_20, 1=FLAT/ROUND_10):", dict(D.die_group_id.value_counts()))

    summary_rows, oof_store = [], {}
    for scheme, splitter in [("leave one RUN out", LeaveOneOut()), ("leave one MATERIAL out", LeaveOneGroupOut())]:
        pY, pF = cross_validate(X, D, Y, F, grid, splitter)
        s, per_run = score(pY, pF, Y, F)
        s.insert(0, "test", scheme)
        summary_rows.append(s)
        oof_store[scheme] = (pY, pF, per_run)

    summary = pd.concat(summary_rows)
    summary.round(1).to_csv(f"{OUT}/cv_scores.csv", index=False)
    print("\nCROSS-VALIDATION SCORES (lower is better):")
    print(summary.round(1).to_string(index=False))

    pY_loro, _, per_run = oof_store["leave one RUN out"]
    pd.DataFrame({
        "run_id": D.run_id, "label": D.label, "die_group_id": D.die_group_id, "total_mm": D.total_mm.round(2),
        **{f"err_pct {m}": (100.0 * per_run[m]).round(1) for m in MODELS},
    }).to_csv(f"{OUT}/per_run_errors.csv", index=False)

    real_models = ["parametric_ridgecv", "parametric_gpr", "pca_ridgecv", "direct_extratrees"]
    strict = summary[(summary.test == "leave one MATERIAL out") & summary.model.isin(real_models)]
    best = strict.sort_values("median_mm_err_pct").model.iloc[0]
    print(f"\nBest ML model under LOMO: '{best}'")

    pred = predict_scenarios_calibrated(best, X, D, Y, F, grid, pY_loro[best])
    pred.round(3).to_csv(f"{OUT}/predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, (name, g) in enumerate(pred.groupby("scenario", sort=False)):
        ax.plot(g.time_s, g.elongation_mm, color=f"C{k}", lw=1.8, label=name)
        ax.fill_between(g.time_s, g.low_mm, g.high_mm, color=f"C{k}", alpha=0.20)
    ax.set_xlabel("time after pressure applied (s)")
    ax.set_ylabel("predicted press movement (mm)")
    ax.set_title(f"Calibrated 5-95% Prediction Intervals ('{best}')")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/predictions.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
