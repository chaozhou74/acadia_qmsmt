# Empirical constants in the timing model — and where to hunt for them in acadia

Every number the visualizer uses is derived from the firmware config or counted out of the
compiled program, **except** the three below. All three were measured on the 4-channel loopback and
none has been traced to its source in acadia yet. This file is the to-do list for that
hunt: what the number is, what it is not, and what evidence any candidate explanation has to
account for.

Both live in `sequence_viz/tracing.py` as named constants, deliberately not folded into a
derived term, so they stay visible as measurements rather than looking like physics.

---

## 1. `MEASURED_BOUNDARY_OFFSET = 1` cycle (5 ns)

**What.** Every blocking `channel_synchronizer` boundary costs one cycle more than
`detect + issue + propagate`, where

| term | source | value |
|---|---|---|
| `detect` | `Acadia._bus_latency("dma_running")` | 3 cycles |
| `issue` | counted from the compiled program: poll instruction through the next `Trigger DMAs` | varies |
| `propagate` | `dma_trigger_dataport` pipelining + 1 for the DMA latching the FIFO output (per the `calculate_trigger_delay` docstring) | 2 cycles |

**Evidence any explanation must fit.**
- +1.000 cycle for one boundary, +2.00 for two, and `four_blocks` compounds correctly at three.
- Independent of DMA push count: 1 / 2 / 3 / 4 pushes all give +1, while `issue` tracks the
  push count exactly (each extra push = exactly +5 ns measured). So the *counting* is right.
- Intra-block layout has **no** offset — back-to-back pulses, `dwell()`, `block=False` FIFO
  batching and barrier alignment padding all agree to ≤0.05 ns. So it is specific to the
  boundary, not to pulses or to the block interior.
- Out-of-sample: fitted on one-boundary/4-push data, then predicted `two_blocks_3ch`,
  `four_blocks` and `barrier_uneven` with no retuning.

**Candidates, none confirmed.**
1. `_bus_latency("dma_running")` is one low — the poll may need a cycle to *act* on the value
   it read, not just to receive it. Look at how `repeat_until`'s hold evaluates `BUS_DATA`
   against `MASK` relative to the bus read's pipeline.
2. The trigger→DMA-load propagation is one low. `calculate_trigger_delay`'s docstring reasons
   about the FIFO output being latched "at the next cycle" — check that against the gateware.
3. A DAC start latency after the DMA loads its command that is not modelled anywhere.

**Why timing alone cannot separate them:** all three are constants entering the same sum. A
non-blocking boundary would isolate `detect`, but a non-blocking boundary has no gap at all,
so there is nothing to compare against.

**Filed as** KI_003.

---

## 2. `MEASURED_BRANCH_PENALTY = 3` cycles (15 ns)

**What.** An edge crossed by a **taken** branch costs three cycles beyond the counted
instruction path, on top of `MEASURED_BOUNDARY_OFFSET`. Applies to a loop back-edge and to a
`test` whose body is skipped forward.

**Evidence.**
- `loop_2` / `loop_3`: 4-push body, 11 counted instructions, measured 20 cycles → X = 3.
- `loop_2_double`: 8-push body, 12 counted instructions, measured 21 cycles → X = 3.
  (8 pushes also drop the FIFO-latency NOPs from 3 to 0, so the body grows by only 1 —
  which the count predicted and the measurement confirmed.)
- `loop_3` intervals are 220.0 and 440.0 ns: exactly linear, so it is per-edge not per-loop.
- The forward `test` skip needed **no** additional constant: poll 107 → branch 108, target 123
  → trigger 130 gives issue 10, and `3 + 10 + 2 + 1 + 3 = 19` cycles = 95 ns, which is what
  `test_false` measured. That the same penalty covers both a backward and a forward taken
  branch is the strongest evidence it is a branch cost rather than a loop cost.

**Candidate.** A pipeline flush on the redirected instruction fetch — three cycles would
suggest a three-stage fetch/decode path. Look for the branch-resolution depth in the
sequencer gateware; if it is not 3, the explanation is wrong.

**Not yet filed separately** — recorded here and in the constant's docstring; see KI_003 for
the related boundary offset.

---

## 3. `MEASURED_CMACC_RESULT_LATENCY = 11` cycles (55 ns)

**What.** A CMACC result ("measurement complete", polled by acadia's `cmacc_done` inside
get_measurement) becomes visible to the sequencer 11 cycles after the end of its capture
window. It only matters after a NON-blocking block: the sequencer then reaches the CMACC poll
while the readout is still playing and waits on it. After a blocking block, the DMA-idle poll
already holds the sequencer until the capture ends and this latency is hidden in the measured
boundary terms. The poll is found from the program (`cmacc_poll_addresses`: a hold whose bus
address is a `module<N>_registers` status register), and the layout then sets the sequencer
clock to max(reached poll, capture end + latency) plus the counted instructions after it.

**Evidence (XM650 loopback, 2026-10-31, GKP FPGA loop with prefetch, forced e).**
- Fitted on the stabilization cycle with a 200-sample forced window: board 3090 ns.
- Same value, no retuning: the logical-measurement cycle (different CD, same window), board
  2450 ns, SeeQuence 2450 ns.
- Out of sample: a 120-sample window (capture ends 400 ns earlier, same instructions): board
  2689.1-2690.0 / 2049.7-2050.7 / 9599.6-9600.2 ns; SeeQuence 2690 / 2050 / 9600 ns.

**Candidates, none confirmed.** The CMACC's accumulate-and-classify pipeline after the last
window sample, plus the status-bit bus latency. Look for the CMACC done-flag pipeline depth in
the gateware.

---

## Not constants — things that look empirical but are not

Worth knowing so nobody "fixes" them:

- **`issue`** is *counted*, not fitted, and its slope has been verified against push count
  (forward edges) and body size (loop edges).
- **The ~283 ns first-arrival latency** is DAC→cable→ADC and is never used by the model. It
  cancels in a within-channel interval, which is why every measurement is an interval.
- **The ~1.5 ns channel-to-channel spread** is real cable/analog difference, measured once
  with the KI_001 dummy channel enabled. Not modelled, and not needed.
- **The 5 ns quantisation** is the capture sample spacing, not a model limit. Averaging 5000
  iterations and interpolating the edge resolves ~0.05 ns.

## How to check a candidate explanation

`timing_validation.py --revalidate` re-derives every case against its measured run. Change a
constant, re-run, and any explanation that is right will hold all 29 cases at ≤0.15 ns — not
just the one it was reasoned from. That is the test that caught the difference between a
constant term and a miscount in the first place.
