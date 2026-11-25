"""
registry_tools.py  -  build registry.csv for the sintering pipeline.

  python registry_tools.py scan       -> walks ROOT, writes registry_draft.csv   (you then edit it by hand)
  python registry_tools.py finalize   -> checks the edited draft, computes density, writes registry.csv
"""
import os, re, sys, math
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>  THE ONLY LINE YOU MUST EDIT BEFORE `scan`  <<<<<<<<<<<<<<<<<<<<
ROOT = r"C:\Users\Lenovo\Metallurgy\Sem_3\Exploratory Project\Sintering Kinetics Combined graphs"
DRAFT, FINAL = "registry_draft.csv", "registry.csv"
COMP = ["Mo_wt", "Fe_wt", "TiB_vol"]            # must match COMP in sintering_pipeline.py


def _num(pattern, text):
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None

def _label(tib, mo, fe):
    """Readable material name used to group runs (edit by hand in the sheet if you prefer other names)."""
    s = "Ti"
    if mo: s += f"-{mo:g}Mo"
    if fe: s += f"-{fe:g}Fe"
    if tib: s += f"-{tib:g}TiB"
    return s


# ------------------------------------------------------------------ scan
def scan(root=ROOT, out=DRAFT):
    rows = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in (".xls", ".txt"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                head = f.read(500).decode("latin-1")
            if not head.startswith("Specimen"):            # skip real Excel files / processed copies / combined files
                continue
            meta = {}
            for line in head.splitlines()[:12]:
                if ":" in line:
                    k, v = line.split(":", 1); meta[k.strip()] = v.strip()
            ctx = os.path.relpath(p, root)                  # folder + file names carry the conditions
            tib = _num(r"(\d+(?:\.\d+)?)\s*(?:vol%?\s*)?TiB", ctx)
            mo, fe = _num(r"(\d+(?:\.\d+)?)\s*Mo", ctx), _num(r"(\d+(?:\.\d+)?)\s*Fe", ctx)
            rows.append(dict(
                run_id=meta.get("Sample ID"), path=p, sheet="",
                label=_label(tib, mo, fe),
                T_C=_num(r"(\d{4})\s*c\b", ctx), P_MPa=_num(r"(\d+)\s*MPa", ctx),
                Mo_wt=mo, Fe_wt=fe, TiB_vol=tib,
                powder_g=None, final_thickness_mm=None, rho_theoretical_gcc=None,
                final_density_rel=None, HV_mean=None,
                skip=1 if "fgm" in ctx.lower() else 0,       # layered (graded) samples: excluded by default
                die_diam_mm=_num(r"(\d+(?:\.\d+)?)", meta.get("Diameter", "")),
                specimen=meta.get("Specimen"), sample_length=meta.get("Sample Length"), width=meta.get("Width"),
                header_name=meta.get("File Name"), date=meta.get("Start Date"), folder=os.path.dirname(ctx)))
    df = pd.DataFrame(rows)
    df["duplicate_id"] = df.run_id.duplicated(keep=False)
    df = df.sort_values(["folder", "run_id"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} raw runs found -> {out}")
    print(f"  {int(df.duplicate_id.sum())} rows share a Sample ID (keep one, set skip=1 on the others)")
    print(f"  {int((df.skip == 1).sum())} rows auto-skipped as FGM")
    print(f"  {int(df.T_C.isna().sum())} rows with no temperature parsed, {int(df.P_MPa.isna().sum())} with no pressure parsed")


# ------------------------------------------------------------------ finalize
def finalize(draft=DRAFT, out=FINAL):
    df = pd.read_csv(draft, dtype={"run_id": str})
    df = df[df.skip.fillna(0) != 1].copy()                    # drop rows you marked skip=1
    df[COMP] = df[COMP].fillna(0)                             # absent element = 0
    # relative density from geometry, only where you did not type a measured value
    need = df.final_density_rel.isna() & df[["powder_g", "die_diam_mm", "final_thickness_mm", "rho_theoretical_gcc"]].notna().all(axis=1)
    area_cm2 = math.pi * (df.die_diam_mm / 20) ** 2
    vol_cm3 = area_cm2 * df.final_thickness_mm / 10
    df.loc[need, "final_density_rel"] = (df.powder_g / vol_cm3 / df.rho_theoretical_gcc)[need]
    print(f"computed final_density_rel for {int(need.sum())} runs from geometry")

    problems = []
    if df.run_id.duplicated().any():
        problems.append(f"duplicate run_id still present: {df.run_id[df.run_id.duplicated()].tolist()}")
    for col in ["label", "T_C", "P_MPa"]:
        bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(bad): problems.append(f"{col} missing for run_id {bad.run_id.tolist()}")
    bad = df[(df.final_density_rel < 0.5) | (df.final_density_rel > 1.02)]
    if len(bad): problems.append(f"final_density_rel outside 0.5-1.02 for run_id {bad.run_id.tolist()}  (units? 96 instead of 0.96?)")
    missing = [p for p in df.path if not os.path.exists(p)]
    if missing: problems.append(f"{len(missing)} file paths do not exist, e.g. {missing[0]}")
    if problems:
        print("\nFIX THESE IN THE DRAFT, THEN RUN finalize AGAIN:")
        for p in problems: print("  -", p)
        return
    df.to_csv(out, index=False)
    print(f"\nOK: {len(df)} runs written to {out}")
    print(df.groupby("label").size().to_string())
    print("\nmissing values per column (blank is fine for hardness/density, they just skip those analyses):")
    print(df[["final_thickness_mm", "final_density_rel", "HV_mean"]].isna().sum().to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "scan": scan()
    elif mode == "finalize": finalize()
    else: print(__doc__)

