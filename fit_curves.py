#!/usr/bin/env python3
"""
fit_curves.py  --  execution_model_v1

Fit empirical response curves from leg_telemetry.parquet and emit:
  - logs/execution_model_v1.json   (lookup tables + fit params for planner)
  - logs/execution_model_v1.txt    (human-readable diagnostic report)

Cohort discipline:
  TRUTH = run_ok & ~sleep_contaminated & ~is_placeholder
          & (prestart_deterministic == True)      # reliability_v1 only
  Leg 0 is modeled separately from non-leg-0.

Curves fit (in order):
  1. v_achieved(v_cmd, leg_len)         non-leg-0 only
  2. launch_proxy(leg_len, cmd_cruise)  leg-0 only
  3. decel_envelope(v_entry, turn_angle)
  4. effective_accel(v_entry_proxy)
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np
import pandas as pd

LOGS = "/Users/conradweeden/ai-grand-prix/logs"
SRC = os.path.join(LOGS, "leg_telemetry.parquet")
OUT_JSON = os.path.join(LOGS, "execution_model_v1.json")
OUT_TXT = os.path.join(LOGS, "execution_model_v1.txt")

# Binning edges (meters, degrees, m/s)
LEG_LEN_EDGES = [0, 8, 12, 16, 20, 24, 30, 45]
V_CMD_EDGES = [0, 5, 7, 8, 9, 9.5, 10, 10.5, 11, 12]
TURN_EDGES = [0, 15, 30, 45, 60, 90, 180]
V_ENTRY_EDGES = [0, 4, 6, 7, 8, 9, 10, 12]

MIN_BIN_N = 8  # minimum rows per bin to report a stable estimate


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "   nan"
    return f"{x:.{n}f}"


def bin_label(edges, i):
    return f"[{edges[i]:g},{edges[i + 1]:g})"


def bin_table(df, x_col, y_col, edges):
    """Return a 1D binned summary dict."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        s = df[(df[x_col] >= lo) & (df[x_col] < hi)][y_col].dropna()
        n = len(s)
        if n == 0:
            out.append(dict(bin=bin_label(edges, i), n=0,
                            mean=None, p10=None, p50=None, p90=None, std=None))
            continue
        out.append(dict(
            bin=bin_label(edges, i),
            n=int(n),
            mean=float(s.mean()),
            p10=float(s.quantile(0.10)),
            p50=float(s.quantile(0.50)),
            p90=float(s.quantile(0.90)),
            std=float(s.std()) if n > 1 else 0.0,
        ))
    return out


def bin_table_2d(df, x_col, y_col, z_col, xe, ye, agg="median"):
    """2D binned median (or mean) of z over (x, y) bins."""
    grid = []
    for i in range(len(xe) - 1):
        row = []
        for j in range(len(ye) - 1):
            sub = df[
                (df[x_col] >= xe[i]) & (df[x_col] < xe[i + 1])
                & (df[y_col] >= ye[j]) & (df[y_col] < ye[j + 1])
            ][z_col].dropna()
            if len(sub) < MIN_BIN_N:
                row.append(dict(n=int(len(sub)), val=None))
            else:
                val = float(sub.median() if agg == "median" else sub.mean())
                row.append(dict(n=int(len(sub)), val=val))
        grid.append(row)
    return grid


def fit_linear(x, y):
    """Return slope, intercept, r2, n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return dict(slope=None, intercept=None, r2=None, n=int(len(x)))
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    syy = ((y - ym) ** 2).sum()
    if sxx == 0:
        return dict(slope=None, intercept=None, r2=None, n=int(len(x)))
    slope = sxy / sxx
    intercept = ym - slope * xm
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else None
    return dict(slope=float(slope), intercept=float(intercept),
                r2=float(r2) if r2 is not None else None, n=int(len(x)))


def fit_quadratic(x, y):
    """y = a x^2 + b x + c, via numpy.polyfit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return dict(a=None, b=None, c=None, r2=None, n=int(len(x)))
    coefs = np.polyfit(x, y, 2)
    a, b, c = float(coefs[0]), float(coefs[1]), float(coefs[2])
    yhat = a * x * x + b * x + c
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return dict(a=a, b=b, c=c,
                r2=float(r2) if r2 is not None else None, n=int(len(x)))


def main():
    if not os.path.exists(SRC):
        print(f"missing: {SRC}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(SRC)
    print(f"loaded: {len(df)} rows, {df['run_uid'].nunique()} runs")

    # --- Cohort discipline ---
    truth = df[
        (df["run_ok"] == True)
        & (df["sleep_contaminated"] == False)
        & (df["is_placeholder"] == False)
        & (df["prestart_deterministic"] == True)
    ].copy()
    print(f"truth cohort: {len(truth)} rows, {truth['run_uid'].nunique()} runs")

    non0 = truth[truth["is_leg0"] == False].copy()
    leg0 = truth[truth["is_leg0"] == True].copy()
    print(f"  non-leg0: {len(non0)} rows")
    print(f"  leg0    : {len(leg0)} rows")

    # Effective acceleration proxy on every leg that has a sustain phase.
    # a_eff = (max_ach - min_ach) / phase_time_sustain_s
    def add_accel_proxy(d):
        denom = d["phase_time_sustain_s"].where(d["phase_time_sustain_s"] > 0.05)
        d["accel_eff"] = (d["max_ach_speed"] - d["min_ach_speed"]) / denom
        return d
    non0 = add_accel_proxy(non0)
    leg0 = add_accel_proxy(leg0)

    report = {}

    # =========================================================
    # 1. v_achieved(v_cmd, leg_len) — non-leg0
    # =========================================================
    # 1a: 2D median grid
    grid = bin_table_2d(non0, "leg_length", "avg_cmd_speed", "avg_ach_speed",
                        LEG_LEN_EDGES, V_CMD_EDGES, agg="median")
    report["v_achieved_grid"] = dict(
        x_col="leg_length", y_col="avg_cmd_speed", z_col="avg_ach_speed",
        x_edges=LEG_LEN_EDGES, y_edges=V_CMD_EDGES, grid=grid,
    )

    # 1b: Global quadratic v_ach = a*v_cmd^2 + b*v_cmd + c
    q = fit_quadratic(non0["avg_cmd_speed"].values, non0["avg_ach_speed"].values)
    report["v_achieved_quadratic_global"] = q

    # 1c: Linear per leg_length bin — slope/intercept of v_ach vs v_cmd
    per_len = []
    for i in range(len(LEG_LEN_EDGES) - 1):
        lo, hi = LEG_LEN_EDGES[i], LEG_LEN_EDGES[i + 1]
        sub = non0[(non0["leg_length"] >= lo) & (non0["leg_length"] < hi)]
        f = fit_linear(sub["avg_cmd_speed"].values, sub["avg_ach_speed"].values)
        f["leg_len_bin"] = bin_label(LEG_LEN_EDGES, i)
        per_len.append(f)
    report["v_achieved_linear_per_leglen"] = per_len

    # 1d: Residuals — underdelivery by v_cmd bin
    ud_by_vcmd = bin_table(non0, "avg_cmd_speed", "underdelivery", V_CMD_EDGES)
    report["underdelivery_by_vcmd_non0"] = ud_by_vcmd

    # =========================================================
    # 2. Launch proxy — leg 0 only
    # =========================================================
    # leg0 response: avg_ach as function of avg_cmd and leg_length
    launch_linear_cmd = fit_linear(leg0["avg_cmd_speed"].values, leg0["avg_ach_speed"].values)
    launch_linear_len = fit_linear(leg0["leg_length"].values, leg0["avg_ach_speed"].values)
    report["launch_linear_avgach_vs_vcmd"] = launch_linear_cmd
    report["launch_linear_avgach_vs_leglen"] = launch_linear_len

    # leg0 binned summary: avg_ach as function of leg_length
    launch_by_len = bin_table(leg0, "leg_length", "avg_ach_speed", LEG_LEN_EDGES)
    report["launch_avgach_by_leglen"] = launch_by_len

    # leg0 binned summary: max_ach as function of leg_length
    launch_max_by_len = bin_table(leg0, "leg_length", "max_ach_speed", LEG_LEN_EDGES)
    report["launch_maxach_by_leglen"] = launch_max_by_len

    # leg0 gate_time as function of leg_length (time to clear first gate)
    gate0_by_len = bin_table(leg0, "leg_length", "gate_time", LEG_LEN_EDGES)
    report["launch_gatetime_by_leglen"] = gate0_by_len

    # =========================================================
    # 3. Decel envelope — pre-turn braking
    # =========================================================
    # Use v_entry proxy = avg_cmd_speed (entry speed into leg's preturn region).
    # We want to know: for a given turn severity, what entry speed is
    # actually survivable? Proxy = avg_ach_speed at bins of (cmd, turn).
    decel_grid = bin_table_2d(non0, "avg_cmd_speed", "turn_severity_deg",
                              "avg_ach_speed",
                              V_ENTRY_EDGES, TURN_EDGES, agg="median")
    report["decel_envelope_grid"] = dict(
        x_col="avg_cmd_speed", y_col="turn_severity_deg", z_col="avg_ach_speed",
        x_edges=V_ENTRY_EDGES, y_edges=TURN_EDGES, grid=decel_grid,
    )

    # Same grid but with preturn_onset_ratio (how far from gate the planner
    # started slowing) — tells us if planner is over-conservative.
    ratio_grid = bin_table_2d(non0, "avg_cmd_speed", "turn_severity_deg",
                              "preturn_onset_ratio",
                              V_ENTRY_EDGES, TURN_EDGES, agg="median")
    report["preturn_onset_ratio_grid"] = dict(
        x_col="avg_cmd_speed", y_col="turn_severity_deg",
        z_col="preturn_onset_ratio",
        x_edges=V_ENTRY_EDGES, y_edges=TURN_EDGES, grid=ratio_grid,
    )

    # Linear regression of underdelivery on turn_severity (is braking
    # the dominant source of speed loss on high-turn legs?)
    report["underdelivery_vs_turn_linear"] = fit_linear(
        non0["turn_severity_deg"].values, non0["underdelivery"].values
    )

    # =========================================================
    # 4. Effective accel
    # =========================================================
    # distribution of accel_eff on non-leg0, binned by avg_ach_speed
    accel_by_vach = bin_table(non0, "avg_ach_speed", "accel_eff", V_ENTRY_EDGES)
    report["accel_eff_by_vach_non0"] = accel_by_vach

    # Distribution on leg0 as well (launch accel)
    accel_leg0_by_len = bin_table(leg0, "leg_length", "accel_eff", LEG_LEN_EDGES)
    report["accel_eff_leg0_by_leglen"] = accel_leg0_by_len

    # Global simple stats
    report["accel_eff_summary"] = dict(
        non0_n=int(non0["accel_eff"].dropna().shape[0]),
        non0_median=float(non0["accel_eff"].median()) if non0["accel_eff"].notna().any() else None,
        non0_p90=float(non0["accel_eff"].quantile(0.90)) if non0["accel_eff"].notna().any() else None,
        leg0_n=int(leg0["accel_eff"].dropna().shape[0]),
        leg0_median=float(leg0["accel_eff"].median()) if leg0["accel_eff"].notna().any() else None,
    )

    # =========================================================
    # 5. Meta
    # =========================================================
    report["meta"] = dict(
        source=SRC,
        rows_total=int(len(df)),
        runs_total=int(df["run_uid"].nunique()),
        rows_truth=int(len(truth)),
        runs_truth=int(truth["run_uid"].nunique()),
        rows_non0=int(len(non0)),
        rows_leg0=int(len(leg0)),
        leg_len_edges=LEG_LEN_EDGES,
        v_cmd_edges=V_CMD_EDGES,
        turn_edges=TURN_EDGES,
        v_entry_edges=V_ENTRY_EDGES,
        min_bin_n=MIN_BIN_N,
        cohort_rule="run_ok & ~sleep_contaminated & ~is_placeholder & prestart_deterministic",
    )

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {OUT_JSON}")

    # ===== Human-readable report =====
    lines = []
    lines.append("execution_model_v1  --  fit report")
    lines.append("=" * 60)
    m = report["meta"]
    lines.append(f"source: {m['source']}")
    lines.append(f"rows total: {m['rows_total']}  runs total: {m['runs_total']}")
    lines.append(f"truth rows: {m['rows_truth']}  truth runs: {m['runs_truth']}")
    lines.append(f"non-leg0: {m['rows_non0']}   leg0: {m['rows_leg0']}")
    lines.append(f"cohort: {m['cohort_rule']}")
    lines.append("")
    lines.append("1. v_achieved(v_cmd, leg_len)  -- non-leg0 truth cohort")
    lines.append("   median avg_ach_speed per (leg_len x v_cmd) bin:")
    header = "   leg_len \\ v_cmd   " + "  ".join(
        f"{bin_label(V_CMD_EDGES, j):>10s}" for j in range(len(V_CMD_EDGES) - 1)
    )
    lines.append(header)
    for i, row in enumerate(report["v_achieved_grid"]["grid"]):
        prefix = f"   {bin_label(LEG_LEN_EDGES, i):>12s}    "
        cells = []
        for cell in row:
            if cell["val"] is None:
                cells.append(f"   -    ({cell['n']:3d})")
            else:
                cells.append(f"{cell['val']:6.2f} ({cell['n']:3d})")
        lines.append(prefix + "  ".join(f"{c:>16s}" for c in cells))
    lines.append("")

    q = report["v_achieved_quadratic_global"]
    lines.append(
        f"   global quadratic fit v_ach = {fmt(q['a'])}*v_cmd^2 + "
        f"{fmt(q['b'])}*v_cmd + {fmt(q['c'])}   R^2={fmt(q['r2'],3)}  n={q['n']}"
    )
    lines.append("   linear v_ach = slope*v_cmd + int  per leg_length bin:")
    for f in report["v_achieved_linear_per_leglen"]:
        lines.append(
            f"     {f['leg_len_bin']:>12s}  slope={fmt(f['slope'])}  "
            f"int={fmt(f['intercept'])}  R^2={fmt(f['r2'],3)}  n={f['n']}"
        )
    lines.append("")

    lines.append("   underdelivery (v_cmd - v_ach) by v_cmd bin, non-leg0:")
    for b in report["underdelivery_by_vcmd_non0"]:
        lines.append(
            f"     v_cmd={b['bin']:>10s}  n={b['n']:4d}  "
            f"p10={fmt(b['p10'])}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}  p90={fmt(b['p90'])}"
        )
    lines.append("")

    lines.append("2. Launch proxy  -- leg0 truth cohort")
    lc = report["launch_linear_avgach_vs_vcmd"]
    ll = report["launch_linear_avgach_vs_leglen"]
    lines.append(
        f"   avg_ach = {fmt(lc['slope'])}*v_cmd + {fmt(lc['intercept'])}  "
        f"R^2={fmt(lc['r2'],3)}  n={lc['n']}"
    )
    lines.append(
        f"   avg_ach = {fmt(ll['slope'])}*leg_len + {fmt(ll['intercept'])}  "
        f"R^2={fmt(ll['r2'],3)}  n={ll['n']}"
    )
    lines.append("   leg0 avg_ach by leg_length:")
    for b in report["launch_avgach_by_leglen"]:
        lines.append(
            f"     {b['bin']:>12s}  n={b['n']:4d}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}  p90={fmt(b['p90'])}"
        )
    lines.append("   leg0 max_ach by leg_length:")
    for b in report["launch_maxach_by_leglen"]:
        lines.append(
            f"     {b['bin']:>12s}  n={b['n']:4d}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}  p90={fmt(b['p90'])}"
        )
    lines.append("   leg0 gate_time by leg_length:")
    for b in report["launch_gatetime_by_leglen"]:
        lines.append(
            f"     {b['bin']:>12s}  n={b['n']:4d}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}  p90={fmt(b['p90'])}"
        )
    lines.append("")

    lines.append("3. Decel envelope / pre-turn behavior")
    lines.append("   median avg_ach per (v_cmd x turn_deg) bin:")
    header = "   v_cmd \\ turn_deg " + "  ".join(
        f"{bin_label(TURN_EDGES, j):>10s}" for j in range(len(TURN_EDGES) - 1)
    )
    lines.append(header)
    for i, row in enumerate(report["decel_envelope_grid"]["grid"]):
        prefix = f"   {bin_label(V_ENTRY_EDGES, i):>12s}    "
        cells = []
        for cell in row:
            if cell["val"] is None:
                cells.append(f"   -    ({cell['n']:3d})")
            else:
                cells.append(f"{cell['val']:6.2f} ({cell['n']:3d})")
        lines.append(prefix + "  ".join(f"{c:>16s}" for c in cells))
    lines.append("")
    lines.append("   median preturn_onset_ratio per (v_cmd x turn_deg) bin:")
    lines.append("   (higher = planner started slowing sooner / more conservative)")
    lines.append(header)
    for i, row in enumerate(report["preturn_onset_ratio_grid"]["grid"]):
        prefix = f"   {bin_label(V_ENTRY_EDGES, i):>12s}    "
        cells = []
        for cell in row:
            if cell["val"] is None:
                cells.append(f"   -    ({cell['n']:3d})")
            else:
                cells.append(f"{cell['val']:6.3f} ({cell['n']:3d})")
        lines.append(prefix + "  ".join(f"{c:>16s}" for c in cells))
    lines.append("")
    ut = report["underdelivery_vs_turn_linear"]
    lines.append(
        f"   underdelivery = {fmt(ut['slope'])}*turn_deg + {fmt(ut['intercept'])}  "
        f"R^2={fmt(ut['r2'],3)}  n={ut['n']}"
    )
    lines.append("")

    lines.append("4. Effective accel proxy = (max_ach - min_ach) / sustain_s")
    s = report["accel_eff_summary"]
    lines.append(
        f"   non-leg0: n={s['non0_n']}  median={fmt(s['non0_median'])}  "
        f"p90={fmt(s['non0_p90'])}"
    )
    lines.append(f"   leg0    : n={s['leg0_n']}  median={fmt(s['leg0_median'])}")
    lines.append("   non-leg0 accel_eff by avg_ach_speed:")
    for b in report["accel_eff_by_vach_non0"]:
        lines.append(
            f"     v_ach={b['bin']:>10s}  n={b['n']:4d}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}"
        )
    lines.append("   leg0 accel_eff by leg_length:")
    for b in report["accel_eff_leg0_by_leglen"]:
        lines.append(
            f"     {b['bin']:>12s}  n={b['n']:4d}  p50={fmt(b['p50'])}  "
            f"mean={fmt(b['mean'])}"
        )
    lines.append("")

    report_txt = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(report_txt)
    print(report_txt)


if __name__ == "__main__":
    main()
