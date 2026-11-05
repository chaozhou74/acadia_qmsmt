# Timing model

How the tracer decides when each DMA command plays, and where every number comes from.
Times are sequencer cycles (5 ns at 200 MHz). Nothing here is fitted: the sequencer and
DMA behaviour is read off the gateware (`acadia/logic/src/acadia_sequencer.vhd`,
`acadia_dma.vhd`, and the bus modules `pyacadia/acadia/hdl.py` generates), and the
pipeline settings come from the firmware configuration. Values fixed in the HDL are
constants in `tracing.py`; values that depend on the configuration are computed by
`tracing.gateware_timing`.

## Sequencer

- **One instruction per cycle** on a straight path.
- **A taken branch costs 3 more cycles** (`TAKEN_BRANCH_CYCLES`). After a branch the PC
  register updates, the instruction memory reads the target (one-cycle read), and the
  instruction register loads it before it executes. This applies to a loop's back-edge and
  to a `test` whose body is skipped.
- **A poll is a PC hold.** `repeat_until` with an empty body compiles to one instruction
  that re-executes every cycle while its condition holds. If it held at least once, the
  next instruction executes 3 cycles after the cycle the condition reads clear
  (`HOLD_EXIT_CYCLES`: the fetch restarts as after a branch, but the PC already points past
  the hold). If the condition is already clear the first time the hold executes, it costs
  one cycle like any other instruction.

## DMA

- **Trigger to first sample: 4 cycles.** The sequencer registers the bus write, the trigger
  dataport registers it again, the DMA loads the descriptor on that edge, and the first
  sample leaves its output register one cycle later (`DMA_START` = load + output register).
- **A playing channel continues without a trigger.** On the last cycle of a descriptor the
  DMA pulls the next one if the FIFO holds one, so it plays without a gap. A trigger that
  arrives while the channel is playing is ignored. A channel whose FIFO ran empty stops and
  waits for the next trigger.
- **Push to pullable: 5 cycles.** The sequencer and the bus decoder each register the FIFO
  write, then the FIFO's own write-to-read latency (`dma_fifo_latency` in the firmware
  configuration, 3).
- **`dma_running` is read low 1 cycle after the playout ends.** `running` drops one cycle
  before the last sample leaves the output register; the running dataport and the bus
  decoder's read register delay it by one cycle each.
- **`fifo_empty` / `fifo_almost_empty` are read 1 cycle after the descriptor that changed
  them starts playing.** The flag changes when a descriptor is pulled, 2 cycles before it
  plays; the FIFO flag, the DMA status register and the decoder each register it once.
  `almost_empty` means one descriptor is still waiting in the FIFO, so it changes one
  descriptor earlier than `fifo_empty`.

## What follows from it

- **A blocking boundary costs `issue + 6` cycles** of dead time when the poll waited.
  `issue` is counted out of the compiled program: the instructions from the poll through the
  next `Trigger DMAs`, along the path that actually executes, plus 3 per taken branch. The 6
  is 1 (status read) + 3 (hold exit) + 4 (trigger) − 2 (the poll and the trigger are counted
  in `issue`). A block with one pulse gives `issue` 6, i.e. 60 ns.
- **A poll that is already satisfied when it first executes costs 2 cycles less.** For a
  blocking block this happens only when the block ends within one cycle of its start (a
  one-cycle `dwell` joining channels), since the poll executes 6 instructions after the
  trigger.
- **After a `block=False` drain** the next block's time follows from the same hold rule,
  with the FIFO flag in place of `dma_running`; consecutive polls are holds in program order.
- **Non-blocking blocks get no gap**: the sequencer only spends the instructions up to the
  next trigger, and the channels play their FIFOs back to back.

## Not modelled

- The DAC and ADC paths add a fixed latency that is the same for every channel; it cancels
  in every interval and is left out.
- The ~1.5 ns channel-to-channel spread on the loopback is cable and analog difference.
- `Acadia._bus_latency` sets how many NOPs the compiler puts before a hold. Those are
  counted as instructions like any other; they do not change when the hold reads.

## Checking it

`validation/timing_validation.py --revalidate` re-traces every loopback case against its
measured run. A gateware or firmware-configuration change that moves any of the numbers
above shows up there as whole-cycle errors at the boundaries it affects.
