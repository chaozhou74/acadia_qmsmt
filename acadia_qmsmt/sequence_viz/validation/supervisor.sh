#!/bin/bash
# Continuous loopback verification. Runs until killed; survives session limits and restarts.
#
# Board work needs no API session -- it is Python plus the board -- so this keeps measuring
# through anything that interrupts the assistant. Everything it runs is RESUMABLE: novel_cases.py
# skips signatures already measured, so a restart continues rather than repeating.
#
# EDITING THIS WHILE IT RUNS: bash reads a script incrementally and remembers a byte offset,
# so rewriting the file in place makes the running copy resume mid-line in whatever now sits
# at that offset. Write a new file and `mv` it over this one (rename swaps the inode; the
# running shell keeps reading the old one), or restart the loop -- every step is resumable.
#
# The ordering is deliberate: highest information first.
#   1. novel signatures  -- one deploy per distinct MODEL PATH, the rest proven redundant
#   2. control flow      -- repeat_until/test across counts, both arms, both nesting directions
#   3. wider structures  -- deeper nesting and longer batches, where novelty is still being found
#   4. re-verify         -- the whole archive re-scored offline against the current model
# Then it loops, because a model change makes every past measurement worth re-scoring.
set -u
cd /home/boson/acadia_qmsmt/acadia_qmsmt/sequence_viz
PY=/home/boson/acadia_env/bin/python
export PYTHONPATH=/home/boson/acadia/pyacadia
LOG=/home/boson/acadia_qmsmt/acadia_qmsmt/sequence_viz/validation/supervisor.log

say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# One thing on the board at a time. Two deploys at once corrupt each other's capture, and a
# stale run from a previous supervisor would do exactly that.
wait_for_board(){
  while pgrep -f "timing_validation.py --(all|case|cases|scan|pairs|triples)" >/dev/null 2>&1 \
     || pgrep -f "novel_cases.py --deploy" >/dev/null 2>&1; do
    sleep 30
  done
}

round=0
while true; do
  round=$((round + 1))
  say "=== round $round ==="

  say "novel signatures: deploying one representative per distinct model path"
  wait_for_board
  $PY -u validation/novel_cases.py --deploy 2>&1 | grep -vE "^(DEBUG|INFO)" | tee -a "$LOG"

  say "control flow: repeat_until and test across counts, arms and nesting directions"
  for spec in "repeat_until_op:loop_count=1,2,3,4,5,6,8" \
              "test_in_counter_loop:loop_count=2,3,4,5" \
              "counter_loop_in_test:loop_count=2,3,4" \
              "test_nested:test_register_value=1,7" \
              "counter_loop_in_test:test_register_value=1,7" \
              "test_in_counter_loop:test_register_value=1,7"; do
    wait_for_board
    say "  scan $spec"
    $PY -u validation/timing_validation.py --scan "$spec" 2>&1 \
      | grep -E "worst|FAIL|scan worst" | tee -a "$LOG"
  done

  say "re-scoring every archived loopback run against the current model"
  $PY -u validation/timing_validation.py --revalidate 2>&1 \
    | grep -E "worst across|MISMATCH" | tee -a "$LOG"

  say "re-tracing archived runs against their own compiled.log (breadth-first)"
  $PY -u validation/stress_campaign.py archive 300 2>&1 \
    | grep -E "ok=|Traced" | tee -a "$LOG"

  say "checking the panel is a pure function of (folder, point, pins), not of click order"
  QT_QPA_PLATFORM=offscreen $PY -u validation/path_independence.py 8 2>&1 \
    | grep -E "routes to|PROBLEM|FAIL" | tee -a "$LOG"

  # A fresh window of interleavings every round: repeating seeds 0..5 only re-proves what
  # already passed, and every storm bug so far came from an interleaving nobody had run.
  say "drawing invariants at a dense sample of sweep points, not the usual six"
  $PY -u validation/sweep_points.py 10 --points 30 2>&1 \
    | grep -E "sweep points over|PROBLEM" | tee -a "$LOG"

  say "random event storms on interleavings nobody has run yet"
  QT_QPA_PLATFORM=offscreen $PY -u validation/gui_robustness.py \
    --seeds 4 --from $((30 + round * 4)) 2>&1 \
    | grep -E "problem\(s\) total|STUCK|UNGUARDED" | tee -a "$LOG"

  say "the controls that move you -- jump list, both scrollbars -- land where they say"
  QT_QPA_PLATFORM=offscreen $PY -u validation/navigation.py 6 2>&1 \
    | grep -E "navigation moves|PROBLEM" | tee -a "$LOG"

  say "what the tooltip says about the bar under the cursor, at both zoom units"
  $PY -u validation/hover_truth.py 8 2>&1 \
    | grep -E "tooltip readings|PROBLEM|NOTE" | tee -a "$LOG"

  say "every canvas size and both themes -- pixel rules and colour rebinding"
  $PY -u validation/render_geometry.py 6 2>&1 \
    | grep -E "renders over|PROBLEM" | tee -a "$LOG"

  say "walking the viewport from the whole sequence to below the floor, and past it"
  $PY -u validation/zoom_extremes.py 8 2>&1 \
    | grep -E "viewport states|PROBLEM" | tee -a "$LOG"

  say "round $round complete; sleeping 5 min before the next"
  sleep 300
done
