#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# day1_dcl.sh — first-validation script when the DCL SDK arrives
# ─────────────────────────────────────────────────────────────────
#
# Expected run order on the morning the DCL AI Race League Python API
# drops (~May 2026). Derived from docs/DCL_INTEGRATION_CHECKLIST.md §3.
# Each step is isolated so a failure at step N doesn't obscure what
# steps 1..N-1 produced.
#
# Usage:
#   scripts/day1_dcl.sh [--model PATH_TO_YOLO_WEIGHTS]
#
# Precondition: fill in `DCLSimAdapter` so its hot-path methods no
# longer raise NotImplementedError. Run `test_dcl_adapter_seam.py`
# first — it will FAIL on the flipped stub (expected). Invert those
# tests to their "works" form, then re-run this script.
#
# Everything is logged to ./dcl_day1_logs/<UTC-timestamp>/ for forensic
# comparison with the mock_kinematic baseline. Don't delete the logs
# until the DCL path is green on all four steps — step 4 comparisons
# need step 0.
# ─────────────────────────────────────────────────────────────────

set -uo pipefail  # -e OFF on purpose: we want every step to attempt
                  # even if a prior step failed, so the operator sees
                  # the full failure surface on day 1.

# ─── Config ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-src/vision/gate_yolo/models/gate_corners_v1.pt}"
COURSE="${COURSE:-technical}"
TIMEOUT_S="${TIMEOUT_S:-30}"

# Parse --model override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --course) COURSE="$2"; shift 2 ;;
        --timeout) TIMEOUT_S="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -40
            exit 0
            ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="dcl_day1_logs/${STAMP}"
mkdir -p "$LOG_DIR"

STATUS_FILE="$LOG_DIR/STATUS.txt"
: > "$STATUS_FILE"

# ─── Helpers ─────────────────────────────────────────────────────
banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}

log_result() {
    # $1 = step name, $2 = rc, $3 = log file tail
    local name="$1" rc="$2" logf="$3"
    if [[ "$rc" == "0" ]]; then
        echo "  ✓ PASS  $name" | tee -a "$STATUS_FILE"
    else
        echo "  ✗ FAIL  $name  (rc=$rc, log: $logf)" | tee -a "$STATUS_FILE"
    fi
}

# ─── Step 0: hermetic sanity ─────────────────────────────────────
# Confirms the mock_dcl chain still works before we touch DCL. If this
# fails, the race stack itself is broken and any DCL failure downstream
# would be confusing noise. Takes ~15 s.
banner "Step 0/4 — mock_dcl hermetic smoke test (baseline sanity)"
LOG0="$LOG_DIR/step0_mock_dcl_smoke.log"
python test_dcl_smoke.py > "$LOG0" 2>&1
RC0=$?
log_result "mock_dcl smoke test" "$RC0" "$LOG0"
if [[ "$RC0" != "0" ]]; then
    echo "  [abort hint] race stack broke independent of DCL — fix "
    echo "  test_dcl_smoke.py failures before touching DCL."
fi

# ─── Step 1: adapter state + command seam ────────────────────────
# VirtualDetector projects from known gates, so this run stresses
# ONLY get_state() + send_velocity_ned() on the DCL adapter. If it
# passes, the pose/telemetry plumbing is correct end-to-end.
# If it fails: coordinate frame flip, unit conversion, or async wrapping.
banner "Step 1/4 — DCL backend, virtual detector (state + command seam)"
LOG1="$LOG_DIR/step1_dcl_virtual.log"
python run_race.py \
    --backend dcl \
    --detector virtual \
    --course "$COURSE" \
    --timeout "$TIMEOUT_S" \
    > "$LOG1" 2>&1
RC1=$?
log_result "dcl + virtual detector completes $COURSE" "$RC1" "$LOG1"
if [[ "$RC1" != "0" ]]; then
    echo "  [debug hint] check DCLSimAdapter.get_state() coordinate "
    echo "  frame (NED expected) and send_velocity_ned() sign + yaw units."
fi

# ─── Step 2: camera seam ────────────────────────────────────────
# YoloPnp consumes adapter.get_camera_frame(). If step 1 passes and
# step 2 fails, the regression is in the camera path — shape, dtype,
# BGR vs RGB, or a missing axis transpose.
banner "Step 2/4 — DCL backend, YOLO detector (camera seam)"
LOG2="$LOG_DIR/step2_dcl_yolo.log"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  ⊘ SKIP  yolo_pnp step — model file not found at $MODEL_PATH" \
        | tee -a "$STATUS_FILE"
    echo "            Set MODEL_PATH=... or pass --model PATH." \
        | tee -a "$STATUS_FILE"
    RC2=SKIP
else
    python run_race.py \
        --backend dcl \
        --detector yolo_pnp \
        --course "$COURSE" \
        --model-path "$MODEL_PATH" \
        --timeout "$TIMEOUT_S" \
        > "$LOG2" 2>&1
    RC2=$?
    log_result "dcl + yolo_pnp detector completes $COURSE" "$RC2" "$LOG2"
    if [[ "$RC2" != "0" ]]; then
        echo "  [debug hint] sandbox-inspect a single frame before "
        echo "  debugging the race loop:"
        echo "      python -c 'import asyncio; from sim.adapter import make_adapter; "
        echo "                  a = make_adapter(\"dcl\"); asyncio.run(a.connect()); "
        echo "                  f = asyncio.run(a.get_camera_frame()); "
        echo "                  print(f.shape, f.dtype, f[0,0])'"
    fi
fi

# ─── Step 3: IMU / fusion seam ──────────────────────────────────
# Only relevant once get_imu() is wired. Catches specific-force vs
# gravity-subtracted acceleration confusion (see checklist §5 item 4).
banner "Step 3/4 — DCL backend + --fusion (IMU + ESKF seam)"
LOG3="$LOG_DIR/step3_dcl_fusion.log"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  ⊘ SKIP  fusion step — model file not found at $MODEL_PATH" \
        | tee -a "$STATUS_FILE"
    RC3=SKIP
else
    python run_race.py \
        --backend dcl \
        --detector yolo_pnp \
        --model-path "$MODEL_PATH" \
        --course "$COURSE" \
        --fusion \
        --vision-pos-sigma 1.0 \
        --timeout "$TIMEOUT_S" \
        > "$LOG3" 2>&1
    RC3=$?
    log_result "dcl + yolo + fusion completes $COURSE" "$RC3" "$LOG3"
    if [[ "$RC3" != "0" ]]; then
        echo "  [debug hint] run the IMU stationary sanity check:"
        echo "      python -c 'import asyncio; from sim.adapter import make_adapter; "
        echo "                  a = make_adapter(\"dcl\"); asyncio.run(a.connect()); "
        echo "                  r = asyncio.run(a.get_imu()); "
        echo "                  print(\"accel:\", r.accel_body, \"(should be ~[0,0,-9.81])\")'"
        echo "  If accel ≈ [0,0,0], DCL publishes gravity-subtracted "
        echo "  acceleration, not specific force. Add gravity back in the adapter."
    fi
fi

# ─── Step 4: bench comparison (MANUAL — bench isn't backend-agnostic yet) ──
banner "Step 4/4 — bench_fusion_ab.py DCL comparison (deferred)"
echo "  ⊘ DEFERRED  bench_fusion_ab.py is currently hardwired to "
echo "               MockKinematicAdapter (see bench_fusion_ab.py line 69)." \
    | tee -a "$STATUS_FILE"
echo "               Day-2 work: add a --backend flag so honest_passes/"
echo "               max_err can be compared DCL vs mock_kinematic at" \
    | tee -a "$STATUS_FILE"
echo "               matched seeds." | tee -a "$STATUS_FILE"
echo "               For now, compare race completion times (run_race.py" \
    | tee -a "$STATUS_FILE"
echo "               output) against the mock_kinematic baseline (~13 s" \
    | tee -a "$STATUS_FILE"
echo "               technical, ~23 s mixed) as a rough sanity check." \
    | tee -a "$STATUS_FILE"

# ─── Summary ─────────────────────────────────────────────────────
banner "Day-1 summary"
cat "$STATUS_FILE"
echo ""
echo "Logs: $LOG_DIR"
echo ""

# Overall exit code: 0 if steps 0 + 1 both pass (step 0 catches stack
# regressions that'd confuse all downstream results; step 1 is the
# minimum viable DCL integration). Steps 2 and 3 can skip cleanly.
if [[ "$RC0" == "0" && "$RC1" == "0" ]]; then
    echo "✓ DCL minimum-viable integration GREEN."
    exit 0
else
    echo "✗ DCL minimum-viable integration RED. Fix step 0 or step 1 first."
    exit 1
fi
