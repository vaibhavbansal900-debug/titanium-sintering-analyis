"""
make_figures_time_fixed.py - Three-phase verification plots (temperature & time) using the
exact same shared read_raw() and segment_run() as sintering_pipeline.py so phase_summary.csv
and run_table.csv are 100% consistent.

Run:  python make_figures_time_fixed.py
Output: outputs/phase_check_fixed/ (4 figures + phase_summary.csv)
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sintering_pipeline import segment_run, REGISTRY

OUTDIR = "outputs/phase_check_fixed"
os.makedirs(OUTDIR, exist_ok=True)
BLUE, RED, GREEN = "#4C78A8", "#E45756", "#54A24B"


def main():
    reg = pd.read_csv(REGISTRY, dtype={"run_id": str})
    runs, rows = [], []
    for _, row in reg.iterrows():
        try:
            seg = segment_run(row)
        except Exception as e:
            print("SKIPPED", row.run_id, "->", e)
            continue
        T, Ts, x, h0, on, end = seg["T"], seg["Ts"], seg["x"], seg["h0"], seg["on"], seg["end"]
        dx_heat = float(x[on] - x[h0])
        dx_hold = float(x[end] - x[on])       # matches total_shrink_mm in run_table.csv identically
        cool = slice(end + 30, len(T))
        ok = (len(T) - end - 30) > 300
        slope = r2 = dx_cool = np.nan
        if ok:
            a = np.polyfit(T[cool], x[cool], 1)
            fit = np.polyval(a, T[cool])
            denom = ((x[cool] - x[cool].mean()) ** 2).sum()
            r2 = float(1.0 - ((x[cool] - fit) ** 2).sum() / denom) if denom > 0 else np.nan
            slope, dx_cool = float(-a[0] * 1000.0), float(x[-1] - x[end])
        note = "no cooling recorded" if not ok else ("cooling record looks wrong - check file" if r2 < 0.95 else "")
        if seg["seg_flags"]:
            note = (note + "; " + "; ".join(seg["seg_flags"])).strip("; ")
        rows.append(dict(
            run_id=row.run_id, label=row.label, T_at_load_C=round(float(Ts[on]), 1),
            dx_heating_mm=dx_heat, dx_pressed_mm=dx_hold, dx_cooling_mm=dx_cool,
            heating_pct_of_pressed=100.0 * dx_heat / dx_hold if dx_hold > 0 else np.nan,
            cooling_slope_um_per_C=slope, cooling_R2=r2, note=note,
        ))
        runs.append(dict(row=row, T=T, x=x, h0=h0, on=on, end=end, ok=ok))

    S = pd.DataFrame(rows)
    S.round(3).to_csv(f"{OUTDIR}/phase_summary.csv", index=False)

    for mode, fname, xlabel in [
        ("T", "1_all_runs_displacement_vs_temperature.png", "temperature (C)"),
        ("t", "4_all_runs_displacement_vs_time.png", "elapsed time (s; 1 Hz sampling)"),
    ]:
        n, ncols = len(runs), 6
        nrows = max(1, math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), runs):
            ax.axis("on")
            T, x, h0, on, end = r["T"], r["x"], r["h0"], r["on"], r["end"]
            xaxis = T if mode == "T" else np.arange(len(x), dtype=float)
            rel = x - x[h0]
            ax.plot(xaxis[h0:on + 1], rel[h0:on + 1], color=BLUE, lw=1.4)
            ax.plot(xaxis[on:end + 1], rel[on:end + 1], color=RED, lw=1.6)
            ax.plot(xaxis[end:], rel[end:], color=GREEN, lw=1.4)
            row = r["row"]
            ax.set_title(f"{row.run_id} {row.label}\n{row.T_C:.0f} C / {row.P_MPa:.0f} MPa", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.grid(alpha=0.25)
        fig.supxlabel(xlabel, fontsize=10)
        fig.supylabel("press movement since 600 C (mm)", fontsize=10)
        fig.legend([plt.Line2D([], [], color=c, lw=2) for c in (BLUE, RED, GREEN)],
                   ["heating (no pressure)", "under pressure (densification)", "cooling"], loc="upper center", ncol=3, fontsize=10)
        plt.tight_layout(rect=(0.01, 0.01, 1, 0.96))
        plt.savefig(f"{OUTDIR}/{fname}", dpi=115)
        plt.close()

    bad = set(S.run_id[S.note.str.contains("wrong", na=False)])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in runs:
        if not r["ok"] or r["row"].run_id in bad:
            continue
        T, x, end = r["T"], r["x"], r["end"]
        tib = r["row"].TiB_vol > 0
        ax.plot(T[end + 30:], x[end + 30:] - x[end + 30], color="#d95f02" if tib else "#1b9e77", alpha=0.65, lw=1.2)
    good = S.dropna(subset=["cooling_slope_um_per_C"])
    good = good[good.cooling_R2 >= 0.95]
    ax.plot([], [], color="#d95f02", label="runs with TiB")
    ax.plot([], [], color="#1b9e77", label="Ti and Ti-Fe-Mo alloys")
    ax.set_xlabel("temperature while cooling (C)")
    ax.set_ylabel("extra press movement since cooling started (mm)")
    ax.set_title(f"Cooling: median slope = {good.cooling_slope_um_per_C.median():.2f} um/C ({len(good)} valid runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/2_cooling_is_equipment.png", dpi=130)
    plt.close()

    Sg = S.sort_values(["label", "run_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(Sg)), 4.8))
    w = 0.27
    xs = np.arange(len(Sg))
    ax.bar(xs - w, Sg.dx_heating_mm.abs(), w, color=BLUE, label="heating (no pressure)")
    ax.bar(xs, Sg.dx_pressed_mm, w, color=RED, label="under pressure")
    ax.bar(xs + w, Sg.dx_cooling_mm.abs().where(~Sg.note.str.contains("wrong", na=False)), w, color=GREEN, label="cooling")
    ax.set_yscale("log")
    ax.set_ylabel("press movement (mm, log scale)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(Sg.run_id, Sg.label)], rotation=90, fontsize=6.5)
    ax.set_title("Press movement in each phase, per run")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/3_movement_by_phase.png", dpi=130)
    plt.close()
    print(f"Saved {len(S)} runs -> {OUTDIR}/")


if __name__ == "__main__":
    main()
