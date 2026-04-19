#!/usr/bin/env python3
"""
harvest_leg_telemetry.py

Walk the non-flight jsonl result files in logs/ and emit one row per (run, leg)
into leg_telemetry.parquet. Also emits a validator report.

Sources scanned:
  - overnight_sweep.jsonl
  - reliability_v1.jsonl
  - ab_9_50_vs_55.jsonl
  - mixed_prestart_validation.jsonl
  - rebaseline_prestart.jsonl

One row per (source, run_idx_in_file, leg_idx). Failed runs get a single
placeholder row with run_ok=False so failure patterns survive into analysis.
"""
from __future__ import annotations
import json, os, sys, glob, math, random
from datetime import datetime, timezone
from collections import defaultdict

LOGS_DIR = "/Users/conradweeden/ai-grand-prix/logs"
OUTPUT_PARQUET = os.path.join(LOGS_DIR, "leg_telemetry.parquet")
OUTPUT_VALIDATOR = os.path.join(LOGS_DIR, "leg_telemetry_validator.txt")

SOURCES = [
    "overnight_sweep.jsonl",
    "reliability_v1.jsonl",
    "ab_9_50_vs_55.jsonl",
    "mixed_prestart_validation.jsonl",
    "rebaseline_prestart.jsonl",
]

# Sleep contamination: flag any run whose inter-run gap (this run's arm_time -
# previous run's arm_time + previous run's total_time) exceeds this many seconds.
# Normal cadence in the overnight sweep was 60-70s; anything >180s is suspicious.
SLEEP_GAP_S = 180.0

# Startup cleanliness thresholds
MAX_ARM_ATTEMPTS_CLEAN = 1
MAX_ALT_SETTLE_CLEAN = 10.0


def safe(d, *keys, default=None):
    """Nested dict getter."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def load_runs(path: str):
    """Yield (row_idx, parsed_json) for a jsonl file."""
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                print(f"[warn] {os.path.basename(path)}:{i} parse failed: {e}", file=sys.stderr)


def build_rows_for_source(path: str):
    """Return list of row dicts for every leg in every run of this source."""
    src = os.path.splitext(os.path.basename(path))[0]
    runs = list(load_runs(path))
    if not runs:
        return []

    # First pass: collect arm_times to detect sleep contamination.
    # Build a list of (row_idx, arm_time, total_time) sorted by arm_time.
    timing = []
    for ridx, r in runs:
        arm_t = safe(r, "pre_start", "arm_time") or 0.0
        tot = r.get("time")
        timing.append((ridx, arm_t, tot))
    sorted_timing = sorted([t for t in timing if t[1] > 0], key=lambda x: x[1])
    sleep_flagged = set()  # set of row_idx
    for i in range(1, len(sorted_timing)):
        prev_idx, prev_arm, prev_tot = sorted_timing[i - 1]
        cur_idx, cur_arm, _ = sorted_timing[i]
        prev_end = prev_arm + (prev_tot or 0.0)
        gap = cur_arm - prev_end
        if gap > SLEEP_GAP_S:
            # Flag the current run (the one that came back after the gap).
            sleep_flagged.add(cur_idx)

    rows = []
    for ridx, r in runs:
        # --- run-level context ---
        arm_t = safe(r, "pre_start", "arm_time") or 0.0
        arm_ts_iso = (            datetime.fromtimestamp(arm_t, tz=timezone.utc).isoformat()
            if arm_t > 1_000_000_000
            else None
        )
        patch_version = safe(r, "pre_start", "patch_version")
        arm_attempts = safe(r, "pre_start", "arm_attempts")
        offboard_attempts = safe(r, "pre_start", "offboard_attempts")
        offboard_health_ok = safe(r, "pre_start", "offboard_health_ok")
        alt_settle_duration = safe(r, "pre_start", "alt_settle_duration")
        hover_dwell = safe(r, "pre_start", "hover_dwell")

        startup_clean = (
            arm_attempts is not None
            and arm_attempts <= MAX_ARM_ATTEMPTS_CLEAN
            and offboard_health_ok is True
            and alt_settle_duration is not None
            and alt_settle_duration <= MAX_ALT_SETTLE_CLEAN
        )
        prestart_deterministic = patch_version is not None
        sleep_contaminated = ridx in sleep_flagged
        completed = bool(r.get("completed"))
        run_ok = completed and (r.get("time") is not None)

        run_ctx = dict(
            source=src,
            run_idx=ridx,
            run_uid=f"{src}:{ridx}",
            overall_run=r.get("overall_run"),
            block=r.get("block"),
            block_run=r.get("block_run"),
            config=r.get("config"),
            version=r.get("version"),
            course=r.get("course"),
            cruise=r.get("cruise"),
            completed=completed,
            run_ok=run_ok,
            total_time=r.get("time"),
            max_spd=r.get("max_spd"),
            avg_cmd_run=r.get("avg_cmd"),
            avg_ach_run=r.get("avg_ach"),
            avg_err_run=r.get("avg_err"),
            util=r.get("util"),
            gates_passed=r.get("gates_passed"),
            accel=r.get("accel"),
            decel=r.get("decel"),
            ceiling=r.get("ceiling"),
            mst=r.get("mst"),
            pr=r.get("pr"),
            util_str=r.get("util_str"),
            util_turn=r.get("util_turn"),
            arm_time=arm_t if arm_t > 0 else None,
            arm_ts_iso=arm_ts_iso,
            patch_version=patch_version,
            arm_attempts=arm_attempts,
            offboard_attempts=offboard_attempts,
            offboard_health_ok=offboard_health_ok,
            alt_settle_duration=alt_settle_duration,
            hover_dwell=hover_dwell,
            startup_clean=startup_clean,
            prestart_deterministic=prestart_deterministic,
            sleep_contaminated=sleep_contaminated,
        )

        legs = r.get("legs") if isinstance(r.get("legs"), list) else []

        if not legs:
            # Emit a single placeholder row so failed runs aren't lost.
            row = dict(run_ctx)
            row.update(
                leg_index=None,
                leg_length=None,
                turn_angle_deg=None,
                gate_time=None,
                leg_time=None,
                phase_time_launch_s=None,
                phase_time_sustain_s=None,
                phase_time_turn_s=None,
                phase_frac_launch=None,
                phase_frac_sustain=None,
                phase_frac_turn=None,
                avg_cmd_speed=None,
                avg_ach_speed=None,
                max_ach_speed=None,
                min_ach_speed=None,
                speed_error_avg=None,
                entry_dist_to_gate=None,
                exit_dist_to_gate=None,
                preturn_onset_dist=None,
                preturn_onset_ratio=None,
                is_leg0=None,
                underdelivery=None,
                underdelivery_ratio=None,
                turn_severity_deg=None,
                is_placeholder=True,
            )
            rows.append(row)
            continue

        for leg in legs:
            phase_t = leg.get("phase_time_s") or {}
            phase_f = leg.get("phase_fraction") or {}
            avg_cmd = leg.get("avg_cmd_speed")
            avg_ach = leg.get("avg_ach_speed")
            underdelivery = (
                (avg_cmd - avg_ach) if (avg_cmd is not None and avg_ach is not None) else None
            )
            underdelivery_ratio = (
                (avg_ach / avg_cmd)
                if (avg_cmd is not None and avg_ach is not None and avg_cmd > 0)
                else None
            )
            turn_ang = leg.get("turn_angle_deg")
            row = dict(run_ctx)
            row.update(
                leg_index=leg.get("leg_index"),
                leg_length=leg.get("leg_length"),
                turn_angle_deg=turn_ang,
                gate_time=leg.get("gate_time"),
                leg_time=leg.get("leg_time"),
                phase_time_launch_s=phase_t.get("LAUNCH"),
                phase_time_sustain_s=phase_t.get("SUSTAIN"),
                phase_time_turn_s=phase_t.get("TURN"),
                phase_frac_launch=phase_f.get("LAUNCH"),
                phase_frac_sustain=phase_f.get("SUSTAIN"),
                phase_frac_turn=phase_f.get("TURN"),
                avg_cmd_speed=avg_cmd,
                avg_ach_speed=avg_ach,
                max_ach_speed=leg.get("max_ach_speed"),
                min_ach_speed=leg.get("min_ach_speed"),
                speed_error_avg=leg.get("speed_error_avg"),
                entry_dist_to_gate=leg.get("entry_dist_to_gate"),
                exit_dist_to_gate=leg.get("exit_dist_to_gate"),
                preturn_onset_dist=leg.get("preturn_onset_dist"),
                preturn_onset_ratio=leg.get("preturn_onset_ratio"),
                is_leg0=(leg.get("leg_index") == 0),
                underdelivery=underdelivery,
                underdelivery_ratio=underdelivery_ratio,
                turn_severity_deg=abs(turn_ang) if turn_ang is not None else None,
                is_placeholder=False,
            )
            rows.append(row)
    return rows


def write_outputs(rows):
    try:
        import pandas as pd
    except ImportError:
        print("[fatal] pandas not available. Install with: pip3 install pandas pyarrow",
              file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows)
    # Try parquet; fall back to feather; then csv.gz.
    written = None
    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        written = OUTPUT_PARQUET
    except Exception as e:
        print(f"[warn] parquet write failed ({e}); trying feather", file=sys.stderr)
        try:
            alt = OUTPUT_PARQUET.replace(".parquet", ".feather")
            df.to_feather(alt)
            written = alt
        except Exception as e2:
            print(f"[warn] feather write failed ({e2}); writing csv.gz", file=sys.stderr)
            alt = OUTPUT_PARQUET.replace(".parquet", ".csv.gz")
            df.to_csv(alt, index=False, compression="gzip")
            written = alt

    # Validator report
    lines = []
    lines.append(f"harvest_leg_telemetry validator")
    lines.append(f"wrote: {written}")
    lines.append(f"rows: {len(df)}")
    lines.append(f"unique runs: {df['run_uid'].nunique()}")
    lines.append("")
    lines.append("rows by source:")
    for src, n in df.groupby("source").size().sort_values(ascending=False).items():
        uniq = df[df["source"] == src]["run_uid"].nunique()
        lines.append(f"  {src:35s}  rows={n:5d}  runs={uniq:4d}")
    lines.append("")
    lines.append("rows by (source, course):")
    for (src, crs), n in df.groupby(["source", "course"]).size().items():
        lines.append(f"  {src:35s} {str(crs):10s}  rows={n}")
    lines.append("")

    lines.append("run_ok x sleep_contaminated:")
    ct = df.drop_duplicates("run_uid").groupby(["run_ok", "sleep_contaminated"]).size()
    for (ok, sc), n in ct.items():
        lines.append(f"  run_ok={ok} sleep_contaminated={sc}  runs={n}")
    lines.append("")

    lines.append("patch_version distribution (per-run):")
    pv = df.drop_duplicates("run_uid").groupby("patch_version", dropna=False).size()
    for k, n in pv.items():
        lines.append(f"  {str(k):25s}  runs={n}")
    lines.append("")

    real_legs = df[~df["is_placeholder"]]
    lines.append(f"real leg rows: {len(real_legs)}")
    lines.append("")

    lines.append("phase-time null rates (real legs only):")
    for c in ["phase_time_launch_s", "phase_time_sustain_s", "phase_time_turn_s"]:
        rate = real_legs[c].isna().mean() if len(real_legs) else float("nan")
        lines.append(f"  {c:25s} null_rate={rate:.3f}")
    lines.append("")

    lines.append("speed null rates (real legs only):")
    for c in ["avg_cmd_speed", "avg_ach_speed", "max_ach_speed", "min_ach_speed",
              "underdelivery", "underdelivery_ratio"]:
        rate = real_legs[c].isna().mean() if len(real_legs) else float("nan")
        lines.append(f"  {c:25s} null_rate={rate:.3f}")
    lines.append("")

    lines.append("leg_length, turn_angle_deg summary (real legs):")
    for c in ["leg_length", "turn_angle_deg", "avg_cmd_speed", "avg_ach_speed",
              "underdelivery", "underdelivery_ratio"]:
        if c in real_legs and len(real_legs):
            s = real_legs[c].dropna()
            if len(s):
                lines.append(
                    f"  {c:22s} n={len(s):5d} min={s.min():.3f} p50={s.median():.3f} "
                    f"mean={s.mean():.3f} max={s.max():.3f}"
                )
    lines.append("")

    lines.append("3 sample runs (first leg only, for spot-checking):")
    sample_uids = random.sample(list(df["run_uid"].unique()),
                                min(3, df["run_uid"].nunique()))
    for uid in sample_uids:
        sub = df[df["run_uid"] == uid].iloc[0]
        keys = ["source", "run_idx", "course", "version", "total_time",
                "run_ok", "sleep_contaminated", "patch_version",
                "leg_index", "leg_length", "turn_angle_deg",
                "avg_cmd_speed", "avg_ach_speed", "underdelivery"]
        lines.append(f"  {uid}")
        for k in keys:
            lines.append(f"    {k} = {sub.get(k)}")
    lines.append("")

    report = "\n".join(lines)
    with open(OUTPUT_VALIDATOR, "w") as f:
        f.write(report)
    print(report)
    return written


def main():
    all_rows = []
    for name in SOURCES:
        path = os.path.join(LOGS_DIR, name)
        if not os.path.exists(path):
            print(f"[skip] missing: {path}")
            continue
        n_before = len(all_rows)
        all_rows.extend(build_rows_for_source(path))
        print(f"[ok] {name}: +{len(all_rows) - n_before} rows")
    if not all_rows:
        print("no rows harvested — check sources")
        sys.exit(1)
    write_outputs(all_rows)


if __name__ == "__main__":
    main()
