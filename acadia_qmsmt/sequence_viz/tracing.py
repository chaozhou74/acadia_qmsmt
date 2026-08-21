"""
Trace the pulse sequence a :class:`QMsmtRuntime` compiles.

Every timing-relevant operation -- ``schedule_pulse``, ``dwell``, the ADC capture
inside ``measure`` -- funnels through ``Acadia.channel_synchronizer``. On exiting
a synchronizer block, ``DMASynchronizer.merge_schedules`` performs the barrier
reconciliation and produces a per-channel, temporally-ordered command list with
lengths already in cycles.

We hook that function rather than reimplementing its timing model, so the
rendered timeline is the timeline the hardware plays. ``merge_schedules``
extends the subschedule lists in place, so reading its ``schedules`` argument
back after the call yields the subschedule structure *including* the alignment
dwells acadia inserted -- which is what makes barriers and padding visible.
"""
import re
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from acadia.system import DMASynchronizer
from acadia.sequencer import STP, Destination

from .dryrun import (StopDryRun, already_traced, branch_recorder,
                     hardware_stubbed, preserved_runtime_state)

KIND = {
    DMASynchronizer.ARBITRARY_CONTINUED: "ARB_CONT",
    DMASynchronizer.ARBITRARY: "ARB",
    DMASynchronizer.CONSTANT_CONTINUED: "CONST_CONT",
    DMASynchronizer.DWELL: "DWELL",
    DMASynchronizer.DIRECT: "DIRECT",
}

# The compiled program is read through structured STP fields (see `decode_program` and the
# `_is_*` predicates below), not by matching `instruction.pprint()` text.

# DERIVED from acadia_dma.vhd: one clocked stage that Acadia._bus_latency does not count.
#
# A blocking boundary and a FIFO drain both make the sequencer WAIT for a status bit to change,
# and both bits are produced by a synchronous process inside the DMA:
#
#   running_int_proc:  running_int <= '1' on `trigger`, '0' on `descriptor_done and fifo_empty`
#   bus_miso_proc:     master_bus_miso(0..4) <= running_int / fifo_empty / fifo_almost_empty / ...
#
# Both are `process(clk) ... if rising_edge(clk)`, so the observable bit changes one clock AFTER
# the condition that causes it. `Acadia._bus_latency(port)` models only the READ path -- one cycle
# to load the sequencer's bus register, one for the decoder's pipelined MISO, one for the decoder's
# per-device pipeline stage (3 for `dma_running` on CONFIG_200) -- and deliberately says nothing
# about how the peripheral produced the value. It adds exactly this cycle for the datamover
# controllers, with the comment "its MISO is driven in a synchronous process"; acadia_dma drives
# its MISO the same way and gets no such term.
#
# So a wait costs `detect` (the read path) PLUS one cycle for the state register, at every boundary,
# whatever the number of channels or descriptors -- each channel's flag passes through its own
# identical single stage, in parallel. That is precisely the measured signature on the 4-channel
# loopback (validation/timing_validation.py, 2026-07-27): +1.000 cycle for one boundary, +2.00 for
# two, unchanged across 1 / 2 / 4 channels, while intra-block layout is exact to 0.01 cycle.
#
# The dataport's own `"pipeline": 1` is NOT a second stage: the generator emits registers for
# `range(1, pipeline)`, so 1 means "no delay" (acadia/pyacadia/acadia/hdl.py, BusDataport.
# generate_hdl). That is what leaves exactly one uncounted stage rather than two.
#
# The same fact explains why a poll whose condition is ALREADY TRUE costs one cycle less: there is
# no transition to observe, so the state register is not on the path. One firmware stage, charged
# when a change must be waited for and not charged when it must not.
DMA_STATUS_REGISTER = 1

# A boundary crossed by a TAKEN branch (a loop back-edge, or a `test` skipping forward) costs
# three more cycles than the straight-line instruction count, on top of DMA_STATUS_REGISTER.
#
# DERIVED FROM THE FIRMWARE, then confirmed by measurement -- not fitted. acadia_sequencer.vhd's
# instruction_proc shows exactly where the three cycles go when a branch writes the PC:
#
#     pc_wr <= '1';                       -- 1: the cycle that writes the PC
#     instruction_p <= (others => '0');   -- 2: the fetch pipeline's first stage is flushed
#     instruction   <= (others => '0');   -- 3: and its second stage
#     ...
#     elsif(pc_wr = '1') then
#         -- "If the PC was just updated in the previous cycle, we need one more
#         --  cycle of nothing before we can load the output of the memory"
#
# So the cost is the depth of the instruction fetch pipeline (instruction_p -> instruction) plus
# the PC-write cycle itself: 3. That the loopback independently measures 3 is a CHECK on the
# reading of the VHDL, not the origin of the number -- which matters, because a constant fitted
# to the cases someone happened to test generalises only to those cases, while a pipeline depth
# read off the hardware description generalises to every branch the sequencer can take.
#
# The measurements that confirm it (timing_validation.py loop_2 / loop_3 / loop_2_double,
# 2026-07-27): a 4-push body measured 20 cycles against 11 counted instructions, an 8-push body
# 21 against 12 -- the counting tracks each body exactly and only this constant is extra. If the
# firmware's fetch pipeline ever gains or loses a stage, this must follow it.
MEASURED_BRANCH_PENALTY = 3

# `test` condition strings come from dryrun's branch_recorder (a rendering of the Python
# condition object), NOT from the compiled program -- so this stays a small text match.
CONDITION_RE = re.compile(r"^(REG\d+)\s*(==|!=|<=|>=|<|>)\s*(\S+)$")
# A counter-driven `repeat_until(counter == target)`: two operands (register/DSP names or
# literals) around a comparison. Used to resolve how many times such a loop runs.
COUNTER_RE = re.compile(r"^(\w+)\s*(==|!=|<=|>=|<|>)\s*(\w+)$")
# The incrementing counter in that idiom is a DSP (loaded 0, +1 per pass); the other operand
# is the target. `REG <op> literal` waits are NOT this -- their register is the varying side.
COUNTER_NAME_RE = re.compile(r"DSP\d+")


@dataclass
class Command:
    """One DMA command placed on the timeline. Times are in sequencer cycles."""
    channel: str
    kind: str
    start: int
    length: int
    is_padding: bool = False
    symbolic: Optional[str] = None
    #: compile-time value of a symbolic length that is NOT a runtime unknown (barrier padding
    #: built as an acadia Operation). Recorded so a UI can decline to offer it as settable; the
    #: command is deliberately not laid out at this length -- see _build_trace.
    static_length: Optional[int] = None
    resolution: str = "fallback"      # "cache" | "override" | "fallback"
    pulse: Optional[str] = None
    io_name: Optional[str] = None
    address: Optional[int] = None

    @property
    def stop(self):
        return self.start + self.length


@dataclass
class Block:
    """One ``channel_synchronizer`` block."""
    index: int
    start: int
    length: int
    trigger: bool
    blocking: bool
    barriers: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    subschedules: list = field(default_factory=list)   # commands grouped by barrier
    conditional: tuple = ()     # control-flow blocks this sits inside, outermost first
    gap_after: int = 0          # dead cycles before the next block's first sample
    gap_breakdown: dict = field(default_factory=dict)

    @property
    def stop(self):
        return self.start + self.length


@dataclass
class Placement:
    """One *execution* of a block. Duck-types :class:`Block` so renderers need no change.

    A block appears once in ``SequenceTrace.blocks`` (the compiled structure) but once per
    iteration in ``SequenceTrace.placements`` (what actually runs), which is the whole point
    of branch-aware layout: a ``loop(3)`` body is compiled once and executed three times.
    """
    index: int                  # index into SequenceTrace.blocks
    iteration: int              # 0-based pass through the enclosing loop
    #: enclosing pass indices, outermost first -- identifies THIS execution of the block, which
    #: is what a per-execution override is keyed by
    path: tuple = ()
    start: int = 0
    length: int = 0
    trigger: bool = True
    blocking: bool = True
    barriers: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    conditional: tuple = ()
    gap_after: int = 0
    gap_breakdown: dict = field(default_factory=dict)
    stream: bool = False        # this placement is an unrolled cache-pointer pulse stream

    @property
    def stop(self):
        return self.start + self.length


@dataclass
class SequenceTrace:
    """A full traced sequence, ready to render."""
    blocks: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    channel_ios: dict = field(default_factory=dict)
    envelopes: dict = field(default_factory=dict)         # from the pulse config
    loaded_envelopes: dict = field(default_factory=dict)  # from DAC memory
    samples_per_cycle: dict = field(default_factory=dict)
    ns_per_cycle: float = 1.0
    runtime_class: str = ""
    source: str = ""
    point: int = 0
    unresolved: int = 0
    used_saved_qmsmt: bool = None    # None for a live runtime

    snapshots: list = field(default_factory=list)   # one per acadia.run()
    placements: list = field(default_factory=list)   # what executes, loops unrolled
    control_flow: dict = field(default_factory=dict)
    loop_counts: dict = field(default_factory=dict)  # block index -> iterations to draw
    repeat_counts: dict = field(default_factory=dict)  # block index -> resolved repeat_until count
    #: Constructs whose compiled target makes them NON-TERMINATING on hardware: a ``P+1`` counter
    #: loop whose exit target is 0. The body runs at least once and the counter never comes back to
    #: 0, so the board hangs. Kept separate from a user's pinned 0, which is a drawing hypothesis.
    nonterminating: set = field(default_factory=set)
    #: ``{construct key: {enclosing pass path, ...}}`` -- every execution the layout REACHED,
    #: recorded whether or not it drew anything. Enumerating executions from the drawn placements
    #: instead loses exactly the ones set to zero passes: the execution disappears from the panel
    #: and its tab disappears from the diagram, so the setting hides its own control.
    entered_paths: dict = field(default_factory=dict)
    #: Constructs whose value the BOARD decides at runtime -- the count or arm cannot be
    #: established from this run's data. Independent of any override, so the drawing can keep
    #: saying "this is data-dependent" while you try values out. Holds both the canonical
    #: ``(block, depth)`` key and the bare block, for older readers.
    indeterminate: set = field(default_factory=set)
    path_choices: dict = field(default_factory=dict)  # block index -> take test body?
    assumed_paths: set = field(default_factory=set)   # tests we could not decide
    #: ``{channel: cache word}`` for direct DMA commands read from a FIXED cache address --
    #: a replayed single word rather than a walking pointer. See direct_command_words().
    direct_words: dict = field(default_factory=dict)
    #: ``{channel: cache offset}`` where that channel's register-sourced gate words begin,
    #: established from the data during layout (see machine._register_gate)
    register_stream_starts: dict = field(default_factory=dict)
    #: register-driven lengths that resolved to ZERO. acadia emits `length - 1`, so 0 wraps to an
    #: all-ones length field -- 328 us (16-bit) or ~21 s (32-bit) rather than nothing. Recorded so
    #: a sweep point that does this is visible instead of silently drawn as an empty command.
    length_underflows: list = field(default_factory=list)
    unsupported_paths: set = field(default_factory=set)  # KI_004: speculation=False
    gap_terms: dict = field(default_factory=dict)   # detect/propagate, from firmware
    registers: dict = field(default_factory=dict)         # "REG0" -> {source, cache_word}
    register_sources: dict = field(default_factory=dict)  # "REG0" -> cache word
    #: "REG2"/"DSP0" -> the compile-time IMMEDIATE it was initialised with. A pointer
    #: loop (`repeat_until(pointer == final)`) has both of its endpoints here, so its
    #: pass count is arithmetic rather than a guess -- see repeat_until_count.
    register_immediates: dict = field(default_factory=dict)
    cache_base: int = None           # bus address of cache word 0
    register_cycles: dict = field(default_factory=dict)   # resolved for this point
    register_overrides: dict = field(default_factory=dict)  # user-supplied cycles
    register_names: dict = field(default_factory=dict)      # "REG0" -> "t_echo"
    resolve_indeterminate: int = 0
    point_offset: int = 0            # when only one point was captured
    iterations_forced: bool = False
    truncated_points: bool = False
    # A cache-pointer pulse stream (randomized benchmarking): the loop walks a cache region
    # issuing each word straight to the DMA. `stream` holds where the command region starts,
    # which cache word holds the count, the per-pulse period floor, and the channel; the
    # unroll reads the words out of `point_cache` (this point's full cache) and decodes each
    # via `addr_names` (see describe_cache_stream / relayout). None when there is no stream.
    stream: Optional[dict] = None
    addr_names: dict = field(default_factory=dict)   # (channel num, word addr) -> (io, pulse)
    point_cache: dict = field(default_factory=dict)  # this point's full cache, word -> value
    # Phase 2 execution-model layout (sequence_viz/machine.py), run in parallel behind a
    # `drain_blocks` maps a block=False batch's trigger index (nth) to the issue span of its
    # repeat_until(fifo_empty) drain -- the execution-model layout (machine.py) advances the
    # sequencer clock to last-pulled + that boundary gap there.
    drain_blocks: dict = field(default_factory=dict)

    @property
    def n_points(self):
        """Sweep points captured. Every one shares the same compiled schedule."""
        return len(self.snapshots)

    def select_point(self, index):
        """Show sweep point ``index`` -- no re-tracing needed.

        Re-reads the pulse data from that point's snapshot and, if the point's
        cache values imply a different register-driven length, re-runs the layout.
        """
        if self.snapshots:
            if not 0 <= index < len(self.snapshots):
                raise IndexError(
                    f"point {index} out of range (captured {len(self.snapshots)})")
            snapshot = self.snapshots[index]
            self.loaded_envelopes = {
                key: (array[:, 0].astype(np.float64)
                      + 1j * array[:, 1].astype(np.float64)) / INT16_FULL_SCALE
                for key, array in snapshot["memories"].items()}
            self.register_cycles = {
                name: snapshot["cache"][word]
                for name, word in self.register_sources.items()
                if word in snapshot["cache"]}
            self.point_cache = snapshot["cache"]   # full cache, for a cache-pointer stream
        self.point = index + self.point_offset
        self.relayout()
        return self

    def relayout(self):
        """Lay out the sequence as EXECUTED, building :attr:`placements`.

        Split out from tracing for two reasons: a register-driven length is only known once
        a sweep point is chosen, and a loop body is compiled once but executed many times.
        Each executed pass gets its own :class:`Placement` with its own command copies, so a
        loop body appears once per iteration on the timeline.

        The layout is done by the execution model in ``sequence_viz/machine.py``: it runs the
        compiled program through a per-channel DMA-FIFO two-clock model, so ``block=False``
        FIFO batching, cache-pointer streams and re-sync drains all fall out of one engine
        (hardware-validated on the loopback; see validation/). Intra-block subschedule/barrier
        layout, loop unrolling via :meth:`execution_plan`, and the boundary-gap constants are
        shared here.
        """
        from .machine import machine_layout
        return machine_layout(self)

    def _is_stream_command(self, command):
        """True for the direct DMA command of a cache-pointer stream on its channel.

        The whole DMA word is fetched at runtime, so it comes through as symbolic
        ``BUS_DATA``; :attr:`stream` marks which channel that stream drives.
        """
        return bool(self.stream and command.symbolic == "BUS_DATA"
                    and command.channel == self.stream["channel"])

    def _expand_stream(self, command, placement, start):
        """Unroll a cache-pointer pulse stream into one command per played gate.

        The loop plays ``count`` gates, ``count`` read from the cache word the runtime
        computed the pointer bound from; gate ``k`` is ``cache[start_offset + k]``, a DMA
        word decoding to ``addr = word >> 16`` / ``length = (word & 0xFFFF) + 1`` (the same
        ``waveform_dma_command`` packing the static path uses), and ``addr`` names the pulse
        via :attr:`addr_names`. Gates are laid at the per-pulse *period* ``max(length,
        floor)`` -- the fifo-refill floor means short pulses cannot play back-to-back.

        :return: the cursor after the train (its start plus ``count`` period slots).
        """
        meta = self.stream
        cache = self.point_cache or {}
        count = int(cache.get(meta["count_word"], 0))
        floor = int(meta["floor"])
        post_span = int(meta.get("post_span", floor))
        cnum = meta["channel_num"]
        cursor = start
        for k in range(count):
            word = int(cache.get(meta["start_offset"] + k, 0))
            address = word >> 16
            length = (word & 0xFFFF) + 1
            io_name, pulse = self.addr_names.get((cnum, address), (None, None))
            placement.commands.append(replace(
                command, start=cursor, length=length, symbolic=None,
                resolution="cache", pulse=pulse, io_name=io_name, address=address))
            # Gates are spaced by the per-gate period (max of the pulse length and the push-
            # cadence floor). The LAST gate has no next gate to refill for, so its trailing is
            # the counted post-loop span instead -- the loop exit + drain + next block's push/
            # trigger -- which is where the following block actually begins. (Ending at the bare
            # last-pulse stop would merge the next block into the gate; ending a full floor past
            # it, as before, put the next block ~one refill gap too late.)
            trailing = max(length, floor) if k < count - 1 else max(length, post_span)
            cursor += trailing
        return cursor

    @property
    def commands(self):
        """Commands as executed (loops unrolled). Falls back to the compiled structure
        before the first layout."""
        if self.placements:
            return [c for p in self.placements for c in p.commands]
        return [c for b in self.blocks for c in b.commands]

    @property
    def static_commands(self):
        """Commands as COMPILED, each appearing once. This is what ``compiled.log``
        contains, so cross-checks must use it rather than the unrolled view."""
        return [c for b in self.blocks for c in b.commands]

    def evaluate_condition(self, condition):
        """Try to decide a ``test`` condition from the register values of this point.

        Only the simple ``REGn <op> literal`` forms are handled -- which is what
        ``reg.load(cache[k])`` followed by ``test(reg == value)`` produces, and the register
        value is known because the cache is captured per sweep point. Anything else (a
        measurement result, register arithmetic) returns None and the caller falls back to
        ``path_choices`` or to assuming the body runs.
        """
        match = CONDITION_RE.match((condition or "").strip())
        if not match:
            return None
        name, operator, literal = match.group(1), match.group(2), match.group(3)
        value = self.register_cycles.get(name)
        if value is None:
            return None
        try:
            literal = int(literal, 0)
        except ValueError:
            return None
        return {"==": value == literal, "!=": value != literal,
                "<": value < literal, "<=": value <= literal,
                ">": value > literal, ">=": value >= literal}.get(operator)

    def _operand_value(self, token):
        """Resolve a condition operand to an int, or None.

        A literal, or a register pinned by an override / fed from this point's cache.
        Counters (DSPs) and anything unknown return None.
        """
        try:
            return int(token, 0)
        except ValueError:
            pass
        if token in self.register_overrides:
            return int(self.register_overrides[token])
        value = self.register_cycles.get(token)
        if value is not None:
            return int(value)
        # ...and finally a compile-time immediate. `final.load(cache_base + index + rounds)` is
        # a constant the program carries, so a loop bounded by it is not data-dependent at all --
        # it only looked that way because nothing read the constant back.
        value = self.register_immediates.get(token)
        return int(value) if value is not None else None

    def repeat_until_count(self, context):
        """Iterations a counter-driven ``repeat_until`` runs, or None if not resolvable.

        The idiom (DR_RB.py, the RepeatTomo runtimes): a DSP counter loaded 0 and
        incremented +1 per pass, ``repeat_until(DSP_counter == target)`` where ``target`` is
        a register fed from the per-point cache (or pinned via an override) or a literal. The
        counter reaches ``target`` after exactly ``target`` passes, so the drawn count is the
        target's value.

        Only that form resolves: exactly one operand must be the ``DSPn`` counter, and the
        other must resolve to a value. Everything else -- a fifo-empty drain, a countdown or
        wait register compared to a literal (``REG1 == 0``, whose register value is *not*
        statically known so the literal is not the count), a ``<``/``>`` bound -- returns
        None, and the caller falls back to drawing one data-dependent pass.
        """
        if context.get("kind") != "repeat_until":
            return None
        match = COUNTER_RE.match((context.get("condition") or "").strip())
        if not match or match.group(2) != "==":
            return None
        left, right = match.group(1), match.group(3)
        left_dsp = bool(COUNTER_NAME_RE.fullmatch(left))
        right_dsp = bool(COUNTER_NAME_RE.fullmatch(right))
        if left_dsp == right_dsp:            # need exactly one DSP counter operand
            return None
        counter = left if left_dsp else right
        target = self._operand_value(right if left_dsp else left)
        if target is None:
            return None
        # A counter reaches its exit value after (exit - start) increments. The start is 0 for
        # the `DSP.load(0)` idiom, which is why taking the target alone was right for every case
        # that existed -- but a POINTER loop starts at a cache address and ends at that address
        # plus the round count, so the target alone is a ~1.9-million-pass loop and the model gave
        # up and drew one assumed pass instead. resonator_number_measurement's counting rounds are
        # exactly that shape: `length_pointer.load(base + index)`, `final.load(base + index + N)`,
        # `repeat_until(length_pointer == final)` with `pulse_cep()` once per pass. Both endpoints
        # are compile-time immediates, so N is arithmetic.
        start = self.register_immediates.get(counter, 0)
        count = target - int(start)
        if count < 0:
            return None
        if count == 0:
            # NOT zero passes -- this loop never exits. MEASURED on the loopback 2026-08-14:
            # repeat_until_op with loop_count=0 never returns from the board ("Timeout occurred
            # waiting for line", repeating), while 1..8 all measure clean.
            #
            # That measurement also settles what `repeat_until` means. Two models fit every count
            # from 1 upwards -- test-before-body and test-after-body both give N passes for N >= 1 --
            # and they disagree only at 0: testing first would exit immediately, testing after runs
            # the body, increments to 1, finds 1 != 0 and goes round again until the counter wraps.
            # The board hangs, so the body ALWAYS RUNS AT LEAST ONCE and a target of 0 is a
            # non-terminating sequence, not an empty one. Reported through `nonterminating` rather
            # than drawn as a tidy zero-pass body, which is a picture the hardware cannot produce.
            return None
        return count

    def _zero_target(self, context):
        """Is this a counter loop whose exit target resolves to 0 -- i.e. one that never exits?"""
        if context.get("kind") != "repeat_until":
            return False
        match = COUNTER_RE.match((context.get("condition") or "").strip())
        if not match or match.group(2) != "==":
            return False
        left, right = match.group(1), match.group(3)
        left_dsp = bool(COUNTER_NAME_RE.fullmatch(left))
        if left_dsp == bool(COUNTER_NAME_RE.fullmatch(right)):
            return False
        target = self._operand_value(right if left_dsp else left)
        counter = left if left_dsp else right
        if target is None:
            return False
        return target - int(self.register_immediates.get(counter, 0)) == 0

    def execution_plan(self):
        """``[(block index, iteration), ...]`` in the order the sequencer runs them.

        Control flow NESTS, so this expands the block list depth by depth. At each depth,
        consecutive blocks sharing that depth's context form one body -- and a body includes
        everything nested INSIDE it, which is then expanded recursively for every pass. A
        ``loop`` repeats its own deterministic count. A ``repeat_until`` repeats the count
        :meth:`repeat_until_count` resolves from its condition register/literal (recorded in
        :attr:`repeat_counts`); when that can't be resolved the count is data-dependent, so one
        pass is drawn. A user ``loop_counts[first block of the body]`` overrides either.

        Grouping at the CONTEXT'S OWN DEPTH is what makes a nested body repeat with its parent.
        Matching the *innermost* context instead (what this did before 2026-08-11) silently
        dropped nested blocks out of the enclosing body: for feedback cooling inside
        ``cool_modes`` -- ``repeat_until(mode_DSP == 3)`` wrapping {mode swap, then
        ``repeat_until(qubit_DSP == 1)`` around the measure/reset} -- the swap block's body
        stopped at the swap, so the plan drew THREE swaps back to back and then the cooling
        ONCE, in the wrong order. The compiled program has one swap inside the loop; the
        sequence was right and the drawing was wrong.
        """
        self.repeat_counts = {}
        self.indeterminate = set()
        self.entered_paths = {}
        self.nonterminating = set()
        self._resolve_every_construct()
        return self._expand_contexts(list(range(len(self.blocks))), 0, 0)

    def _resolve_every_construct(self):
        """Read each construct's data-derived value out of the run, reachable or not.

        Resolvability is a property of the RUN'S DATA, not of what is currently drawn. Doing it only
        while expanding the plan meant a construct whose enclosing loop was pinned to 0 passes never
        got resolved, and its tab then claimed "the board decides this at runtime" for a count the
        cache pins down exactly -- a false claim of indeterminacy, which is as misleading as the
        false claim of certainty this marker exists to prevent.
        """
        seen = set()
        for index, block in enumerate(self.blocks):
            for depth, context in enumerate(block.conditional or ()):
                level = depth + 1
                key = (self._context_id(context, level), level)
                if key in seen:
                    continue
                seen.add(key)
                # `index` is the first block of this construct's body, which is how every other
                # part of this module names it
                if context.get("kind") == "test":
                    if self.evaluate_condition(context.get("condition")) is None:
                        self.indeterminate.add(self.construct_key(index, level))
                        self.indeterminate.add(index)
                    continue
                if context.get("kind") == "loop":
                    continue                 # a deterministic count, straight from the program
                count = self.repeat_until_count(context)
                if count is None:
                    # a resolvable target of 0 is not "unknown", it is a loop that never exits
                    if self._zero_target(context):
                        self.nonterminating.add(self.construct_key(index, level))
                        self.nonterminating.add(index)
                    self.indeterminate.add(self.construct_key(index, level))
                    self.indeterminate.add(index)
                else:
                    self.repeat_counts[self.construct_key(index, level)] = int(count)
                    self.repeat_counts.setdefault(index, int(count))

    def _expand_contexts(self, members, depth, iteration, path=()):
        """Expand ``members`` (block indices, in address order) below control-flow ``depth``.

        ``iteration`` is stamped on blocks that bottom out here (no context deeper than
        ``depth``), so every placement reports the pass of its own innermost loop.

        ``path`` is the tuple of ENCLOSING pass indices -- (0,) inside the first pass of one outer
        loop, (2, 1) inside the second pass of a loop inside the third pass of another. It exists
        so an override can name ONE EXECUTION of a construct rather than the construct itself.
        That distinction is not cosmetic: an inner active-reset loop is compiled once but runs
        once per outer pass, and how many rounds it takes depends on what the qubit did that
        time. Keying an override by block alone forced every execution to the same count, so
        setting the cooling rounds for one mode silently set them for all of them -- a picture
        that cannot happen on hardware.
        """
        plan, i = [], 0
        while i < len(members):
            index = members[i]
            stack = self.blocks[index].conditional
            if len(stack) <= depth:          # not inside any context at this depth
                plan.append((index, iteration, tuple(path)))
                i += 1
                continue

            context = stack[depth]
            # The body is every consecutive member inside THIS context, nesting included.
            # Identical sibling loops are told apart by the id branch_recorder stamps on each
            # `with` entry (two cool_qubits calls both read `repeat_until(DSP0 == 1)`); when
            # that id is absent -- an older trace -- this falls back to comparing the context
            # by value, which merges such siblings into one body.
            j = i + 1
            while (j < len(members)
                   and len(self.blocks[members[j]].conditional) > depth
                   and self._same_context(self.blocks[members[j]].conditional[depth], context)):
                j += 1
            body = members[i:j]
            first = body[0]

            if context["kind"] == "test" and context.get("speculation") is False:
                # KI_004: with speculation=False the body is placed OUT OF LINE, so address
                # order stops being execution order and the edge costing below does not
                # apply -- measured 25 ns out on the taken arm, and the skipped arm hangs
                # the sequencer outright. Draw the body but flag the path as unmodelled
                # rather than assert a timeline we know is wrong.
                self.assumed_paths.add(first)
                self.unsupported_paths.add(first)
                plan.extend(self._expand_contexts(body, depth + 1, 0))
            elif context["kind"] == "test":
                # explicit choice wins; otherwise try to decide it from the cache;
                # otherwise assume the body runs (and say so via `assumed_paths`)
                # Ask the DATA first, always -- even when a pin is in force. Whether the board
                # decides this arm at runtime is a property of the construct, not of what is being
                # displayed, and it used to be lost the moment you pinned: the tab dropped its "?"
                # and a hypothesis then looked exactly like a measured fact.
                self.entered_paths.setdefault(
                    self.construct_key(first, depth + 1), set()).add(path)
                decided = self.evaluate_condition(context.get("condition"))
                taken = self._path_override(first, depth + 1, path)
                if taken is None:
                    taken = decided
                    if taken is None:
                        taken = True
                        self.assumed_paths.add(first)
                if taken:
                    plan.extend(self._expand_contexts(body, depth + 1, 0, path + (0,)))
            elif context["kind"] == "repeat_until":
                # count from the loop's condition register/literal when resolvable
                # (recorded so the caption can state it); else data-dependent -> one pass.
                # Same here: resolve from the run's own cache whether or not a pin exists, so
                # "the board decides this count" stays visible while you explore other values.
                self.entered_paths.setdefault(
                    self.construct_key(first, depth + 1), set()).add(path)
                # established for every construct up front by _resolve_every_construct
                resolved = self.repeat_counts.get(self.construct_key(first, depth + 1))
                count = self._count_override(first, depth + 1, path)
                if count is None:
                    count = resolved
                if count is None:
                    count = 1
                for pass_index in range(max(int(count), 0)):
                    plan.extend(self._expand_contexts(body, depth + 1, pass_index,
                                                      path + (pass_index,)))
            else:
                # loop: deterministic count, unrolled.
                count = self._count_override(first, depth + 1, path)
                if count is None:
                    count = context.get("count") or 1
                for pass_index in range(max(int(count), 1)):
                    plan.extend(self._expand_contexts(body, depth + 1, pass_index,
                                                      path + (pass_index,)))
            i = j
        return plan

    @staticmethod
    def construct_key(block, depth, path=None):
        """Canonical identity of a control-flow construct, and optionally of ONE execution.

        The first block of the body is not enough. Nested constructs frequently begin at the same
        block -- a mode loop, the cooling round inside it and the active reset inside that all
        start together -- so keying by block alone made three different constructs share one key.
        Three tabs then all read `@11` and editing any of them edited the same thing, which is
        exactly the "I changed one and something else moved" symptom.

        ``depth`` disambiguates them: within one nesting stack a construct is uniquely identified
        by where it sits. It is the ONE-BASED nesting level -- the same number
        ``control_flow_summary`` and ``branch_regions`` report -- because two conventions for the
        same quantity is how a key silently matches nothing and an edit appears to do nothing. ``path`` narrows it further to a single EXECUTION (see
        :meth:`_expand_contexts`); omit it to mean every execution.
        """
        return (int(block), int(depth)) if path is None else (int(block), int(depth), tuple(path))

    def _count_override(self, first, depth, path):
        """The pinned pass count for THIS execution of a construct, or None.

        Looked up most specific first: ``loop_counts[(block, path)]`` pins one execution,
        ``loop_counts[block]`` pins every execution of that construct. Both are supported because
        both are wanted -- "draw this one cooling round as 3" and "draw all of them as 3" are
        different requests, and only the second used to be expressible.
        """
        keys = [self.construct_key(first, depth)]                 # every execution
        if path is not None:
            keys.insert(0, self.construct_key(first, depth, path))  # this execution
            keys.append((first, tuple(path)))                       # legacy, pre-depth keys
        keys.append(first)                                          # legacy, block only
        for key in keys:
            value = self.loop_counts.get(key)
            if value is not None:
                return value
        return None

    def _path_override(self, first, depth, path=None):
        """The pinned arm for a ``test``, or None -- same precedence as :meth:`_count_override`.

        Exists so the layout and the UI cannot disagree about which pin applies. When the panel
        read ``path_choices[block]`` directly while the layout resolved the depth-qualified key,
        a pinned arm displayed as "resolved": the control said the value came from the data when
        it had in fact been set by hand, which is the one thing this panel must never do.
        """
        keys = [self.construct_key(first, depth)]
        if path is not None:
            keys.insert(0, self.construct_key(first, depth, path))
        keys.append(first)                                # legacy, pre-depth keys
        for key in keys:
            if key in self.path_choices:
                return self.path_choices[key]
        return None

    @staticmethod
    def _context_id(context, level):
        """Logical identity of a control-flow construct instance, shared by every block in its body.

        The same rule :meth:`_same_context` applies, as a hashable key: prefer the per-entry ``id``
        acadia puts on the context, and fall back to object identity only when there is none.
        """
        marker = context.get("id")
        return marker if marker is not None else ("anon", level, id(context))

    @staticmethod
    def _same_context(a, b):
        """Is this the same control-flow block instance? Prefer the per-entry id."""
        a_id, b_id = a.get("id"), b.get("id")
        if a_id is not None and b_id is not None:
            return a_id == b_id
        return a == b

    @property
    def length_cycles(self):
        source = self.placements or self.blocks
        return max((b.stop for b in source), default=0)

    @property
    def length_ns(self):
        return self.length_cycles * self.ns_per_cycle

    def register_label(self, name):
        """``REG0`` plus what it is fed from -- ``t_echo = cache[0]`` if aliased."""
        alias = self.register_names.get(name, name)
        source = self.registers.get(name, {}).get("source")
        return f"{alias} = {source}" if source else alias

    def control_flow_summary(self):
        """Describe every control-flow construct, for a UI that lets the user pin it.

        The trace already accepts both overrides -- ``loop_counts[block] = N`` for how many times
        a ``loop``/``repeat_until`` body is drawn, and ``path_choices[block] = True/False`` for
        which arm of a ``test`` runs -- and :meth:`relayout` re-times everything in place. What
        was missing is a way to ENUMERATE them, so a caller can offer one control per construct
        instead of the user having to know block indices.

        Each entry says where its value came from, which is the part that matters when reading a
        picture: ``resolved`` (read out of the captured cache -- trustworthy), ``assumed`` (the
        trace could not tell and picked a default -- treat the drawing as one possibility, not as
        fact), or ``pinned`` (you set it).

        :return: ``[{block, kind, depth, label, count, taken, source, settable}, ...]`` in
            execution order, outermost first.
        """
        # Every EXECUTION of every construct, keyed the way the tabs are. A construct nested in a
        # loop is compiled once and runs once per enclosing pass, and each of those executions is
        # independently settable -- an active-reset loop takes a different number of rounds every
        # time. Listing only the construct hid that: three tabs on the diagram read "@11" and the
        # panel offered one row for all of them.
        executions = {}
        # ...and what ONE MORE PASS of each construct costs, in placements. Counted from the plan,
        # not from the static block list: a body containing loops expands to far more than its block
        # count (83 passes of a 3-block body produced 583 placements, because the body has an inner
        # loop), and the panel derives its pass limit from this number.
        inside, own_passes = {}, {}
        for placement in (self.placements or ()):
            stack = getattr(placement, "conditional", ()) or ()
            path = tuple(getattr(placement, "path", ()) or ())
            for level in range(1, len(stack) + 1):
                context_here = stack[level - 1]
                key = (id(context_here), level)
                executions.setdefault(key, set()).add(path[:level - 1])
                # The BODY cost has to be grouped by the construct, and every block carries its own
                # COPY of the context dicts -- so id() grouped a single block's placements and said
                # the body of a three-block loop was one block. `context["id"]` is the logical
                # identity, the same one _same_context and this method's own dedupe use.
                logical = (self._context_id(context_here, level), level)
                inside[logical] = inside.get(logical, 0) + 1
                own_passes.setdefault(logical, set()).add(
                    path[level - 1] if len(path) >= level else 0)

        entries, seen = [], set()
        for index, block in enumerate(self.blocks):
            for depth, context in enumerate(block.conditional or ()):
                key = context.get("id", (depth, id(context)))
                if key in seen:
                    continue
                seen.add(key)
                # The construct is named by the first block of its body AND its nesting level.
                # ``depth`` here is one-based, matching construct_key / branch_regions: the two
                # conventions used to differ, so the panel wrote (block, 0) while the drawing read
                # (block, 1) and an edit made from the panel silently did nothing to that tab.
                first = index
                level = depth + 1
                kind = context.get("kind")
                # what one extra pass costs, in placements: everything drawn inside this
                # construct divided by how many of its own passes are drawn. Falls back to the
                # static block count when nothing of it is drawn at all.
                instance = (self._context_id(context, level), level)
                drawn_inside = inside.get(instance, 0)
                drawn_passes = max(len(own_passes.get(instance, ())), 1)
                body = (max(drawn_inside // drawn_passes, 1) if drawn_inside else
                        max(sum(1 for other in self.blocks
                                if len(other.conditional or ()) >= level
                                and self._same_context(other.conditional[level - 1], context)), 1))
                # One entry per EXECUTION of this construct. Taken from what the layout REACHED
                # (entered_paths), not from what it drew: an execution pinned to zero passes draws
                # nothing, and reading the drawn placements dropped it from this list -- which
                # removed its row from the panel and its tab from the diagram, leaving no way back.
                reached = self.entered_paths.get(self.construct_key(first, level))
                paths = sorted(reached if reached is not None
                               else executions.get((id(context), level), set()))
                runs = [{"path": path,
                         "key": self.construct_key(first, level, path),
                         "pinned": self._count_override(first, level, path)
                         if kind != "test" else self._path_override(first, level, path)}
                        for path in paths]
                if kind == "test":
                    pinned = self._path_override(first, level)
                    source = ("pinned" if pinned is not None
                              else "assumed" if first in self.assumed_paths else "resolved")
                    # Whether the arm RAN is read off the executed plan, not inferred from
                    # assumed_paths. An assumed test is drawn TAKEN (the tracer's default when it
                    # cannot decide), so "assumed" and "skipped" are not the same thing -- reading
                    # one as the other would caption a drawn body as a skipped one.
                    taken = (pinned if pinned is not None
                             else any(p.index == first for p in (self.placements or ())))
                    entries.append({
                        "block": first, "kind": "test", "depth": level,
                        # the key a caller must write to change this construct and nothing else
                        "key": self.construct_key(first, level),
                        "indeterminate": self.construct_key(first, level) in self.indeterminate,
                        "executions": runs, "body": body,
                        "label": f"test {context.get('condition') or ''}".strip(),
                        "count": None, "taken": taken,
                        "source": source, "settable": True})
                else:
                    pinned = self._count_override(first, level, None)
                    resolved = self.repeat_counts.get(first)
                    if pinned is not None:
                        count, source = int(pinned), "pinned"
                    elif resolved is not None:
                        count, source = int(resolved), "resolved"
                    elif kind == "loop":
                        count, source = int(context.get("count") or 1), "resolved"
                    else:
                        count, source = 1, "assumed"
                    entries.append({
                        "block": first, "kind": kind or "loop", "depth": level,
                        "key": self.construct_key(first, level),
                        "indeterminate": self.construct_key(first, level) in self.indeterminate,
                        "nonterminating": self.construct_key(first, level) in self.nonterminating,
                        "executions": runs, "body": body,
                        "label": f"{kind} x{count}", "count": count, "taken": None,
                        "source": source, "settable": True})
        return entries

    def register_summary(self):
        """Describe every register / length-symbol for a per-register UI control.

        Two kinds appear, distinguished by ``settable``:

        * cache-fed registers resolve themselves from the per-point cache, so
          their value is shown but not settable. One may drive a command length
          (``is_length``, shown in cycles/ns) or only a ``test``/``repeat_until``
          condition (shown as the raw register value -- which, when it is a loop
          target, is the count the loop is drawn for; it follows the sweep point).
        * register/DSP-driven command *lengths* not recoverable from the cache
          (``resolution`` "fallback"/"override") -- the only thing worth setting,
          via :attr:`register_overrides`.

        :return: ``[{name, label, source, resolution, value_cycles, is_length,
            settable}, ...]``, deduped by name, identified registers first.
        """
        # A length the COMPILER already fixed is not a register the user can pin. Barrier
        # padding is built as an acadia Operation (`max(30, 30)`), which arrives here as a
        # symbolic length and used to be offered in the GUI as a settable register -- a control
        # over a value nothing can change, captioned with a raw `Operation(<built-in function
        # max>, ...)` repr. `static_length` marks those; they are excluded.
        resolved = {c.symbolic: (c.resolution, c.length)
                    for c in self.commands
                    if c.symbolic and getattr(c, "static_length", None) is None}

        def entry(name, resolution, cycles, is_length):
            return {"name": name, "label": self.register_label(name),
                    "source": self.registers.get(name, {}).get("source") or "",
                    "resolution": resolution, "value_cycles": cycles,
                    "is_length": is_length,
                    "settable": resolution in ("override", "fallback")}

        entries = {}
        for name, info in self.registers.items():             # cache/device registers
            if name in resolved:                              # also drives a length
                entries[name] = entry(name, *resolved[name], True)
            else:                                             # condition-only register
                resolution = "cache" if info["cache_word"] is not None else "device"
                entries[name] = entry(name, resolution,
                                      self.register_cycles.get(name), False)
        for name, (resolution, cycles) in resolved.items():   # DSP-driven lengths
            if name not in entries:
                entries[name] = entry(name, resolution, cycles, True)
        return list(entries.values())

    def envelope(self, io_name, pulse, source="memory"):
        """Complex waveform for a pulse, in DAC full-scale units.

        ``source="memory"`` is what the FPGA would actually have played at this
        sweep point -- it is read back out of the DAC waveform memory, so it
        carries the swept scale, the detune and the phase. ``source="config"``
        recomputes from the yaml/duplicate config instead, which is the nominal
        pulse and ignores anything ``load_pulse`` overrode at runtime.

        Falls back to the config when the memory was never loaded (all zeros).
        """
        if source == "memory":
            samples = self.loaded_envelopes.get((io_name, pulse))
            if samples is not None and len(samples) and np.abs(samples).max() > 0:
                return samples
        return self.envelopes.get((io_name, pulse))

    @property
    def dead_time_ns(self):
        """Total inter-block dead time -- the boundary gaps from :func:`edge_gap`."""
        return sum(b.gap_after for b in (self.placements or self.blocks)) * self.ns_per_cycle

    def summary(self):
        of_n = f" of {self.n_points}" if self.n_points > 1 else ""
        lines = [
            f"{self.runtime_class}  ({self.source})",
            f"  sweep point {self.point}{of_n}, {len(self.blocks)} synchronizer "
            f"blocks, {len(self.commands)} commands",
            f"  {self.length_ns:.1f} ns on {len(self.channels)} channels: "
            f"{', '.join(self.channels)}",
        ]
        gapped = [b for b in self.blocks if b.gap_after]
        if gapped:
            lines.append(f"  {self.dead_time_ns:.1f} ns dead across "
                         f"{len(gapped)} blocking block boundaries "
                         f"({100 * self.dead_time_ns / max(self.length_ns, 1):.1f}% "
                         f"of the sequence)")
        if self.unresolved:
            how = {"cache": "from cache", "override": "overridden",
                   "fallback": "fallback"}
            shown = {}
            for c in self.commands:
                if c.symbolic:
                    shown[self.register_label(c.symbolic)] = (
                        f"{c.length * self.ns_per_cycle:.0f} ns "
                        f"({how[c.resolution]})")
            lines.append(f"  {self.unresolved} register/DSP-driven length(s): "
                         + ", ".join(f"{n} -> {v}" for n, v in sorted(shown.items())))
        if self.truncated_points:
            lines.append("  point capture stopped at max_points")
        if self.used_saved_qmsmt is False:
            lines.append("  traced against the INSTALLED acadia_qmsmt "
                         "(the folder's saved copy was not used)")
        return "\n".join(lines)

    def to_text(self, max_blocks=None):
        """Per-channel textual timeline, in ns."""
        out = [self.summary(), ""]
        for blk in self.blocks[:max_blocks]:
            out.append(f"=== block {blk.index}  "
                       f"[{blk.start * self.ns_per_cycle:.1f} -> "
                       f"{blk.stop * self.ns_per_cycle:.1f} ns]  "
                       f"trigger={blk.trigger} blocking={blk.blocking}"
                       + (f"  barriers={len(blk.barriers)}" if blk.barriers else ""))
            for ch in self.channels:
                cmds = [c for c in blk.commands if c.channel == ch]
                if not cmds:
                    continue
                out.append(f"  {ch}")
                for c in cmds:
                    length = (f"<{self.register_label(c.symbolic)}>" if c.symbolic
                              else f"{c.length * self.ns_per_cycle:8.1f} ns")
                    out.append(f"    {c.start * self.ns_per_cycle:10.1f} ns  "
                               f"{'pad ' if c.is_padding else 'user'}  "
                               f"{c.kind:11s} {length}  {c.pulse or ''}")
            if blk.gap_after:
                g = blk.gap_breakdown
                out.append(f"  -> {blk.gap_after * self.ns_per_cycle:.1f} ns dead "
                           f"before the next block "
                           f"(detect {g['detect']} + issue {g['issue']} + "
                           f"propagate {g['propagate']} cycles)")
        if max_blocks is not None and len(self.blocks) > max_blocks:
            out.append(f"... {len(self.blocks) - max_blocks} more blocks")
        return "\n".join(out)


def trace_runtime(runtime, point=0, resolve_indeterminate=0, envelopes=True,
                  capture_points=True, max_points=4096, single_iteration=True,
                  resolve_registers=None, register_names=None,
                  allow_instruments=False):
    """Dry-run ``runtime.main()`` and return the :class:`SequenceTrace`.

    The sequence is compiled once, so the schedule is shared by every sweep
    point; only the pulse data and the cache differ. With ``capture_points`` the
    dry run therefore continues through the whole sweep, snapshotting the DAC
    memories and the cache at each ``acadia.run()``, and any point can then be
    selected with :meth:`SequenceTrace.select_point` without tracing again.

    :param point: which point to select for display afterwards.
    :param capture_points: run the whole sweep and snapshot every point. With
        ``False`` the dry run stops at ``point``, which is quicker for one look at
        a huge sweep.
    :param single_iteration: force ``runtime.iterations = 1`` for the dry run.
        Iterations repeat the same points, so this is what keeps the run bounded
        -- the DR-tomography runtime is 96 points but 1.92M runs otherwise. The
        field is restored afterwards.
    :param max_points: snapshot cap; the dry run stops once it is reached.
    :param resolve_registers: ``{"REG0": cycles}`` to pin specific registers,
        overriding the cache. Use for a register fed by something with no static
        value -- a measurement result, say. ``trace.registers`` lists the names
        and where each is fed from.
    :param register_names: ``{"REG0": "t_echo"}`` display aliases.
    :param resolve_indeterminate: fallback cycles for register/DSP-driven lengths
        that are neither overridden nor resolvable from the cache.
    :param envelopes: compute pulse waveforms as well as the schedule.
    :param allow_instruments: let the runtime reach the instrument server during the
        dry run. Off by default, because those calls move real hardware -- see
        :class:`~sequence_viz.dryrun.InstrumentAccessBlocked`.
    """
    from acadia import DataManager

    if already_traced(runtime):
        raise RuntimeError(
            "this runtime has already been traced -- construct a fresh one.\n"
            "Its Acadia holds the compiled sequencer program from the previous "
            "trace, and acadia.compile() compiles every sequencer instance, so a "
            "second pass fails on the stale one. Forcing it through would also be "
            "wrong: waveform memories and registers are allocated afresh each "
            "pass. Use trace_folder(), which builds a new runtime per call, or "
            "select_point() -- one trace already captures every sweep point.")

    raw_blocks = []
    pending_user_ids = []
    snapshots = []
    branch_stack = []
    state = {"runs": 0}

    orig_create = DMASynchronizer.create_schedules
    orig_merge = DMASynchronizer.merge_schedules
    orig_exit = DMASynchronizer.__exit__

    def spy_create(calls):
        schedules = orig_create(calls)
        pending_user_ids.append({id(c) for sub in schedules
                                 for cmds in sub.values() for c in cmds})
        return schedules

    def spy_merge(schedules, indeterminate_types):
        combined = orig_merge(schedules, indeterminate_types)
        # merge_schedules extended `schedules` in place with the alignment
        # dwells, so it now carries the barrier structure as actually compiled
        raw_blocks.append({
            "subschedules": [{ch: list(cmds) for ch, cmds in sub.items()}
                             for sub in schedules],
            "user_ids": pending_user_ids.pop() if pending_user_ids else set(),
            "conditional": tuple(dict(b) for b in branch_stack),
        })
        return combined

    def spy_exit(self, *a):
        before = len(raw_blocks)
        try:
            return orig_exit(self, *a)
        finally:
            if len(raw_blocks) > before:
                raw_blocks[-1]["trigger"] = self._dma_trigger
                raw_blocks[-1]["blocking"] = self._dma_block

    pool = {}          # content hash -> array, so unchanged memories are shared

    def on_run(self, *a, **k):
        if not capture_points:
            if state["runs"] >= point:
                raise StopDryRun()
            state["runs"] += 1
            return
        snapshots.append(_snapshot(runtime, pool) if envelopes
                         else {"memories": {}, "cache": _cache_words(runtime)})
        state["runs"] += 1
        if len(snapshots) >= max_points:
            raise StopDryRun()

    restore_iterations = _MISSING = object()
    if single_iteration and capture_points and hasattr(runtime, "iterations"):
        restore_iterations = runtime.iterations
        runtime.iterations = 1

    # the reads that build the trace (pulse caches, config envelopes) have to
    # happen before the runtime's state is put back, so everything is inside.
    # the defaults below are set inside too, so they get cleaned up with it.
    with preserved_runtime_state(runtime):
        if not hasattr(runtime, "data_manager"):
            runtime.data_manager = DataManager()
        if not hasattr(runtime, "local_directory"):
            runtime.local_directory = "."

        DMASynchronizer.create_schedules = staticmethod(spy_create)
        DMASynchronizer.merge_schedules = staticmethod(spy_merge)
        DMASynchronizer.__exit__ = spy_exit
        try:
            with hardware_stubbed(on_run, runtime=runtime,
                                  allow_instruments=allow_instruments), \
                 branch_recorder(branch_stack):
                try:
                    runtime.main()
                except StopDryRun:
                    pass
        finally:
            DMASynchronizer.create_schedules = staticmethod(orig_create)
            DMASynchronizer.merge_schedules = staticmethod(orig_merge)
            DMASynchronizer.__exit__ = orig_exit
            if restore_iterations is not _MISSING:
                runtime.iterations = restore_iterations

        if not raw_blocks:
            raise RuntimeError(
                "no synchronizer blocks were traced -- main() never reached a "
                "channel_synchronizer, or it stopped earlier than expected")

        trace = _build_trace(runtime, raw_blocks, resolve_indeterminate)
        trace.register_overrides = dict(resolve_registers or {})
        trace.register_names = dict(register_names or {})
        trace.iterations_forced = restore_iterations is not _MISSING
        trace.truncated_points = bool(max_points) and len(snapshots) >= max_points
        if capture_points:
            trace.snapshots = snapshots
            selected = point
        else:
            # only the requested point was reached, so it is the only snapshot
            trace.snapshots = [_snapshot(runtime, pool)] if envelopes else []
            trace.point_offset = point
            selected = 0

    trace.select_point(selected)
    return trace


def _cache_words(runtime):
    """Every sequencer-cache word currently set, by absolute word index."""
    words = {}
    for instance in runtime.acadia.CacheArray.instances:
        array = getattr(instance, "_array", None)
        if array is None:
            continue
        for offset, value in enumerate(np.asarray(array).reshape(-1)):
            words[instance.index + offset] = int(value)
    return words


def _snapshot(runtime, pool):
    """DAC memory contents and cache words as of right now."""
    memories = {}
    for io_name, io in runtime._ios.items():
        if not io.channel.is_dac:
            continue
        for pulse, entry in io._pulse_cache.items():
            memory = entry.get("memory")
            if memory is None or getattr(memory, "__array_interface__", None) is None:
                continue
            array = np.array(memory.array, copy=True)
            digest = (array.shape, array.tobytes())
            memories[(io_name, pulse)] = pool.setdefault(digest, array)
    return {"memories": memories, "cache": _cache_words(runtime)}


@dataclass
class Instr:
    """One compiled STP instruction, read from its fields (not ``pprint()`` text).

    ``d1``/``d2`` are the two destination-slot major names ("BUS_ADDR", "REG", "PC", "DSP_AB",
    "DSP_C", "DSP_CFG", "BUS_DATA", "MASK", "NONE"), with the paired source major in ``s1``/``s2``
    and immediate in ``imm1``/``imm2`` (slot 1 pairs dest1/src1/imm1). ``d1_minor`` is the port
    index -- or, for a PC destination, the branch/hold code.
    """
    i: int
    d1: Optional[str]
    d1_minor: int
    d2: Optional[str]
    d2_minor: int
    s1: Optional[str]
    s2: Optional[str]
    imm1: object
    imm2: object
    conditional: bool
    condition_invert: bool
    op: Optional[str]
    comment: Optional[str]


def decode_program(acadia):
    """The compiled program as :class:`Instr` records, flat across every sequencer instance
    (the order ``pprint()`` walked). Reading the STP fields directly replaces the fragile
    regexes that used to scan ``instruction.pprint()`` text."""
    prog = []
    for sequencer in acadia._sequencer_type.instances:
        for ins in sequencer._compiled_program:
            d1, d2 = ins.dest1, ins.dest2
            prog.append(Instr(
                i=len(prog),
                d1=d1.major.name if d1 is not None else None,
                d1_minor=int(d1.minor) if d1 is not None else 0,
                d2=d2.major.name if d2 is not None else None,
                d2_minor=int(d2.minor) if d2 is not None else 0,
                s1=ins.src1.major.name if ins.src1 is not None else None,
                s2=ins.src2.major.name if ins.src2 is not None else None,
                imm1=ins.imm1, imm2=ins.imm2,
                conditional=bool(ins.conditional),
                condition_invert=bool(ins.condition_invert),
                op=ins.op, comment=ins.comment))
    return prog


def _resolve_imm(imm):
    """An instruction immediate (int / Symbol / referenced instruction) as a concrete int,
    using acadia's own resolver -- e.g. a branch target or a bus address. None if unresolvable."""
    try:
        return int(STP.assemble_imm(imm))
    except Exception:
        return None


def _is_trigger(r):
    """The ``Trigger DMAs`` instruction that starts a block's queued DMAs playing."""
    return r.comment == "Trigger DMAs"


def _is_dma_poll(r):
    """The blocking wait a block exit emits: PC-hold while ``BUS_DATA AND MASK != 0`` (the
    dma_running poll). ``condition_invert`` False is the ``!= 0`` sense."""
    return (r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_HOLD and r.conditional
            and r.s2 == "BUS_DATA" and r.op == "and" and not r.condition_invert)


def _is_drain_poll(r):
    """The wait a ``block=False`` batch's drain emits: the same PC-hold as
    :func:`_is_dma_poll` but in the INVERTED (``fifo_empty``/``almost_empty``) sense."""
    return (r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_HOLD and r.conditional
            and r.s2 == "BUS_DATA" and r.op == "and" and r.condition_invert)


def _is_branch(r):
    """An absolute PC branch (loop back-edge or `test` skip); target is its slot-1 immediate."""
    return r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_BRANCH


def _static_value(value):
    """The compile-time value of an acadia ``Operation``/``Symbol``, or None if runtime-dependent.

    ``Acadia.command_dma`` itself distinguishes these exactly this way (an assigned ``Symbol`` or
    a resolveable ``Operation`` is printed as its value), so this asks the object rather than
    pattern-matching its repr.
    """
    for attribute, check in (("resolveable", None), ("assigned", True)):
        probe = getattr(value, attribute, None)
        if probe is None:
            continue
        try:
            ok = probe() if callable(probe) else probe
            if ok is (True if check is True else ok) and ok:
                return int(value.value())
        except Exception:
            return None
    return None


def _bus_addr(r):
    """The cache/bus address an instruction writes to BUS_ADDR (from an immediate), else None."""
    if r.d1 == "BUS_ADDR" and r.s1 == "IMM":
        return _resolve_imm(r.imm1)
    if r.d2 == "BUS_ADDR" and r.s2 == "IMM":
        return _resolve_imm(r.imm2)
    return None


def _bus_addr_pointer(r):
    """True when this instruction drives BUS_ADDR from a POINTER rather than a literal.

    ``bus_read(pointer)`` compiles to ``BUS_ADDR <- DSP_P``: the address comes out of a DSP that
    walks the cache, so there is no immediate to read. Distinguishing this from "no bus address
    here at all" matters -- see :func:`describe_registers`.

    WHICH DSP drives it is deliberately not reported. The decoded record carries a minor only for
    DESTINATIONS (``d1_minor``/``d2_minor``); on this instruction that is the bus port, not the
    counter. Naming a specific "DSP0" from it would be inventing a fact -- the same class of
    mistake as the wrong device label this predicate exists to remove.
    """
    for dest, src in ((r.d1, r.s1), (r.d2, r.s2)):
        if dest == "BUS_ADDR" and src == "DSP_P":
            return True
    return False


def _bus_data_load(r, dests):
    """If this instruction lands BUS_DATA into one of ``dests`` (a destination major), return
    ``(major, minor)`` for that slot, else None -- e.g. a cache read into ``REG``/``DSP_AB``."""
    if r.s1 == "BUS_DATA" and r.d1 in dests:
        return (r.d1, r.d1_minor)
    if r.s2 == "BUS_DATA" and r.d2 in dests:
        return (r.d2, r.d2_minor)
    return None


def _pulse_address_map(runtime):
    """(channel number, word address) -> (io name, pulse name) for scheduled pulses."""
    names, spc = {}, {}
    for io_name, io in runtime._ios.items():
        if not io.channel.is_dac:
            continue
        width = io.channel.interface_width_bytes
        spc[str(io.channel)] = getattr(io, "_samples_per_cycle", None)
        for pulse, entry in io._pulse_cache.items():
            mem = entry.get("memory")
            if mem is not None and getattr(mem, "_resource", None) is not None:
                names[(io.channel.num(), mem._resource._resource_id // width)] = \
                    (io_name, pulse)
    return names, spc


def _build_trace(runtime, raw_blocks, resolve):
    """Build the structure. Times are filled in by :meth:`SequenceTrace.relayout`,
    which is re-run whenever a point with different register values is selected."""
    acadia = runtime.acadia
    addr_names, spc = _pulse_address_map(runtime)

    channel_ios = {}
    for io_name, io in runtime._ios.items():
        channel_ios.setdefault(str(io.channel), []).append(io_name)

    trace = SequenceTrace(
        ns_per_cycle=1e9 / acadia.sequencer_clock_frequency(),
        runtime_class=type(runtime).__name__,
        channel_ios=channel_ios,
        samples_per_cycle=spc,
        resolve_indeterminate=int(resolve),
    )
    try:
        trace.control_flow = sequencer_control_flow(acadia)
        dataport = acadia._firmware["sequencer_bus"]["dma_trigger_dataport"]
        trace.gap_terms = {
            "detect": acadia._bus_latency("dma_running"),
            "propagate": (max(dataport["pipeline"])
                          + (1 if dataport["bus_pipeline"] else 0) + 1)}
    except Exception:
        pass                    # gaps are an enhancement; never fail the trace
    try:
        trace.registers = describe_registers(acadia)
    except Exception:
        pass
    trace.register_sources = {name: info["cache_word"]
                              for name, info in trace.registers.items()
                              if info["cache_word"] is not None}
    try:
        trace.register_immediates = describe_immediates(acadia)
    except Exception:
        pass                    # a constant nobody could read is the old behaviour, not a failure
    try:
        # where the cache region starts on the bus, so an ABSOLUTE pointer immediate can be
        # turned into a cache WORD index
        trace.cache_base = acadia._firmware.sequencer_bus_decoder["cache"].address().value()
    except Exception:
        pass

    # A cache-pointer pulse stream (randomized benchmarking) is unrolled from the per-point
    # cache in relayout; store the decode map and the stream descriptor here.
    trace.addr_names = addr_names
    try:
        from .machine import drain_block_issue
        trace.drain_blocks = drain_block_issue(acadia)
    except Exception:
        pass                    # the machine layout is opt-in; never fail the trace
    try:
        trace.direct_words = direct_command_words(acadia)
    except Exception:
        trace.direct_words = {}
    try:
        stream = describe_cache_stream(acadia)
        if stream:
            for io in runtime._ios.values():
                if str(io.channel) == stream["channel"]:
                    stream["channel_num"] = io.channel.num()
                    trace.stream = stream
                    break
    except Exception:
        pass                    # the stream unroll is an enhancement; never fail the trace

    for index, raw in enumerate(raw_blocks):
        block = Block(index=index, start=0, length=0,
                      trigger=raw.get("trigger", True),
                      blocking=raw.get("blocking", True),
                      conditional=raw.get("conditional", ()))
        for sub in raw["subschedules"]:
            group = []
            for ch, cmds in sub.items():
                ch_name = str(ch)
                if ch_name not in trace.channels:
                    trace.channels.append(ch_name)
                for c in cmds:
                    # Usually {"length": n, "address": a}. A command whose whole DMA word
                    # is supplied at runtime instead carries {"command": BUS_DATA} and no
                    # length or address at all -- randomized benchmarking picks its
                    # Clifford that way. Nothing about it is knowable off-hardware, so it
                    # is treated as a symbolic length like a register-driven dwell.
                    raw_len = c["length"] if "length" in c else c.get("command")
                    symbolic = length = static_length = None
                    resolved = _static_value(raw_len)
                    if isinstance(raw_len, (int, np.integer)):
                        length = int(raw_len) + (1 if c.get("length_is_minus_one")
                                                 else 0)
                    else:
                        symbolic = str(raw_len)
                        # An Operation/Symbol that resolves at COMPILE time (barrier padding is
                        # built this way: `max(30, 30)`, `max(30, 30) - 30`). Its value is
                        # recorded so the UI can stop offering it as something to "pin" -- it is
                        # not a runtime unknown -- but the command is still NOT given that
                        # length in the layout.
                        #
                        # Laying it out was tried and is wrong: acadia keeps these entries in
                        # its in-memory schedule and does NOT emit a DMA command for all of
                        # them. Resolving them added six 30-cycle dwells that the board never
                        # played, and four archived runs
                        # (DualRailCCZGateTomographyBasisDebug, DRCCZPhaseCalibration) stopped
                        # matching their own compiled.log -- with `only_in_archive` empty, so
                        # the extra commands were purely invented. The archive is the authority.
                        static_length = resolved
                    io_name = pulse = None
                    address = c.get("address")
                    if address is not None:
                        io_name, pulse = addr_names.get(
                            (ch.num(), address), (None, None))
                    group.append(Command(
                        static_length=static_length,
                        channel=ch_name,
                        kind=KIND.get(c["command_type"], str(c["command_type"])),
                        start=0, length=length, symbolic=symbolic,
                        is_padding=id(c) not in raw["user_ids"],
                        pulse=pulse, io_name=io_name, address=address))
            block.subschedules.append(group)
            block.commands.extend(group)
        trace.blocks.append(block)

    _compute_config_envelopes(runtime, trace)
    return trace


def sequencer_control_flow(acadia):
    """Trigger/poll addresses and branches, for costing an executed edge.

    Straight-line edges can be costed by counting instructions between the poll and the
    next trigger. A loop back-edge cannot: execution leaves the poll, reaches the branch,
    jumps to the branch target, and runs forward from there to that block's trigger. So the
    branch and its target address are needed as well.

    :return: ``{"triggers": [addr], "polls": {nth trigger: addr}, "branches": [(addr, target)],
                "back_branches": [(branch addr, target addr)]}``
    """
    prog = decode_program(acadia)

    triggers = [r.i for r in prog if _is_trigger(r)]
    polls = {}
    for nth, trigger in enumerate(triggers):
        limit = triggers[nth + 1] if nth + 1 < len(triggers) else len(prog)
        # Either kind of wait ends a block: the blocking `dma_running` poll, or the INVERTED
        # `fifo_empty`/`almost_empty` poll that drains a block=False batch. Only the blocking one
        # used to be recorded, so a drain block had no poll here and edge_gap could not cost the
        # edge leaving it -- it returned None, and the layout silently fell back to the
        # address-order span from drain_block_issue. That is correct only when control falls
        # straight through; inside a loop the next executed block is the loop head, reached
        # backwards over the branch, and the address-order span overcharges by one cycle every
        # pass (batch_in_loop measured exactly 5.00 ns x (loop_count - 1)).
        poll = next((j for j in range(trigger + 1, limit)
                     if _is_dma_poll(prog[j]) or _is_drain_poll(prog[j])), None)
        if poll is not None:
            polls[nth] = poll

    # Both directions matter: a loop back-edge jumps backward, while a `test` whose
    # condition fails jumps FORWARD past the skipped body. Either way the executed path
    # leaves the address order and has to be costed in two segments. The target is the
    # branch's own slot-1 immediate, so there is no risk of picking up a different
    # instruction's address (the old text form quoted two spellings and had to guess).
    branches = []
    for r in prog:
        if _is_branch(r):
            target = _resolve_imm(r.imm1)
            if target is not None:
                branches.append((r.i, target))

    return {"triggers": triggers, "polls": polls, "branches": branches,
            "back_branches": [(b, t) for b, t in branches if t < b]}


def _branch_is_taken(at, target, goal, triggers, executed, arms):
    """Would the sequencer take this branch on the path it actually ran?

    Decided from the program and from which blocks executed -- never from the branch's own
    condition, which is a runtime value the compiled program does not carry.

    ``arms`` is a mutable list of "did it run?" flags for the trigger-less `test` arms on this
    stretch, in program order. A `channel_synchronizer(trigger=False)` arm emits no DMA trigger,
    so its branch skips a range containing no trigger at all and there is nothing in the program
    to match it against; the flags come from the execution plan instead and are consumed in the
    order the walk meets them, which is program order for both.
    """
    if target < at:                       # backward: a loop back edge
        return goal < at                  # the goal is behind us, so the edge is the only way
    if at < goal < target:
        return False                      # taking it would jump OVER the block we are going to
    skipped = [t for t in triggers if at < t < target]
    if not skipped:
        if arms:
            return not arms.pop(0)        # a queued arm: taken exactly when it did not run
        return False                      # skips no block at all: not a body-skip branch
    return not any(t in executed for t in skipped)


def _executed_path(control_flow, poll, goal, executed_nths, arms=()):
    """Instructions FETCHED walking from ``poll`` to ``goal``, and the branch penalty paid.

    Counting an address range is only right while control stays in address order, and it does
    not: a `test` arm whose condition fails is jumped over, and a loop takes its back edge. The
    old code handled that by counting one range and then subtracting the one skipped body it
    could find between the poll and the FIRST branch -- so a CHAIN of skipped arms (the prep
    selector: one `test(sel == i)` per prep state, at most one taken) had every skip after the
    first counted as though it had run. Measured on the loopback with test_chain, arm 0 taken:

        arms      1     2     4     8
        measured  365   390   440   540   ns      <- 25 ns per skipped arm, flat
        old model 365   390   520   780   ns      <- +40 ns per skipped arm beyond the first

    Walking the path has no such blind spot: every skip costs its own condition plus one taken
    branch and nothing for the body it jumps over, which is what the hardware does and why the
    measured slope does not care how big the arm is.

    Returns ``(fetched, penalty)``, or ``None`` if the walk does not reach the goal -- in which
    case the caller keeps the address-order count rather than trusting a partial walk.
    """
    triggers = control_flow["triggers"]
    branch_at = dict(control_flow["branches"])
    executed = {triggers[n] for n in (executed_nths or ()) if 0 <= n < len(triggers)}
    limit = 4 * (max(triggers, default=0) + len(branch_at) + 2)
    fetched = penalty = 0
    pc = poll
    pending = list(arms)                         # consumed in program order by the walk
    for _ in range(limit):
        if pc == goal:
            return fetched + 1, penalty          # the goal instruction is fetched too
        fetched += 1
        target = branch_at.get(pc)
        if target is not None and _branch_is_taken(pc, target, goal, triggers, executed,
                                                   pending):
            pc = target
            penalty += MEASURED_BRANCH_PENALTY
        else:
            pc += 1
    return None


def edge_gap(control_flow, from_nth, to_nth, executed_nths=(), arms=()):
    """Dead cycles between two executed blocks, and how it breaks down.

    ``from_nth``/``to_nth`` index the compiled program's trigger list. A backward edge picks
    up the branch instructions and the taken-branch penalty.

    :param executed_nths: trigger indices of the blocks that DO execute. Used to tell a skipped
        `test` body (jumped over, costing nothing but a branch penalty) from one that runs.
    """
    triggers, polls = control_flow["triggers"], control_flow["polls"]
    if from_nth not in polls:
        return None                                  # non-blocking: no wait, no gap
    if not (0 <= to_nth < len(triggers)):
        return None
    poll = polls[from_nth]

    if to_nth == from_nth + 1:
        issue = triggers[to_nth] - poll + 1
        penalty = 0
        kind = "fall-through"
        # ...unless trigger-less `test` arms sit on this stretch. Then control does leave address
        # order even though the next ANCHORED block is the very next one: each skipped arm's
        # branch is taken, jumping over pushes the straight count would charge for. Measured on
        # the loopback (test_chain, trigger=False): 25 ns per skipped arm, against the 15 ns a
        # plain address count gives -- the branch is paid for and its 2 pushes are not.
        if arms and not all(arms):
            walked = _executed_path(control_flow, poll, triggers[to_nth], executed_nths, arms)
            if walked is not None:
                issue, penalty = walked
                kind = f"fall-through, {sum(1 for a in arms if not a)} queued arm(s) skipped"
    else:
        # The executed path leaves address order: find the branch that redirects to a
        # point from which the target block's trigger is reached by running forward.
        # Backward for a loop back-edge, forward for a skipped `test` body.
        candidates = [(b, target) for b, target in control_flow["branches"]
                      if b > poll and target <= triggers[to_nth]]
        if not candidates:
            return None
        branch, target = min(candidates, key=lambda pair: pair[0])
        # poll .. branch inclusive, then branch target .. that block's trigger inclusive
        issue = (branch - poll + 1) + (triggers[to_nth] - target + 1)
        penalty = MEASURED_BRANCH_PENALTY
        kind = ("taken branch (loop back)" if target < branch
                else "taken branch (skip)")

        # Prefer WALKING the executed path over counting address ranges: the two agree while
        # control stays in order and part ways as soon as more than one body is skipped on the
        # way (see _executed_path). The address count stays as the fallback for a walk that
        # cannot reach the target, so an unfamiliar program shape degrades to the old answer
        # instead of to a wrong one.
        walked = _executed_path(control_flow, poll, triggers[to_nth], executed_nths, arms)
        if walked is not None:
            issue, penalty = walked
            skips = penalty // MEASURED_BRANCH_PENALTY
            kind = ("taken branch (loop back)" if target < branch
                    else "taken branch (skip)")
            if skips > 1:
                kind += f" x{skips}"
            return {"issue": issue, "branch_penalty": penalty, "kind": kind}

        # A SKIPPED BLOCK on this stretch was counted as executed. The run above is a straight
        # instruction count from the poll to `branch`, but if a `test` body between them was
        # skipped, the sequencer jumped over those instructions and never paid for them -- while
        # paying one more taken-branch penalty for the skip itself. Correct both.
        #
        # This is the feedback-cooling shape (a `test`/`repeat_until(feedback)` inside the round
        # loop), which 53 of the qudit runtimes build via cool_modes/cool_qubits and which no
        # case exercised until `test_in_loop_false`. Measured on the loopback: the uncorrected
        # count is 11 phantom instructions minus the missing 3-cycle penalty = 8 cycles = 40 ns
        # long, against a measured error of 39.95 ns; corrected it predicts 110 ns and measures
        # 110.05 ns. Only branches whose whole skipped span lies inside (poll, branch) are
        # corrected, so a single-branch edge is untouched -- verified identical on loop_2,
        # loop_3, loop_2_double, test_true, test_false and both nested_cool cases.
        for skip_at, skip_to in control_flow["branches"]:
            if not (poll < skip_at < branch and skip_at < skip_to <= branch):
                continue                              # not a forward skip inside this stretch
            if not any(skip_at < t < skip_to for t in triggers):
                continue                              # skips no block: nothing was jumped over
            if any(skip_at < triggers[n] < skip_to for n in executed_nths or ()):
                continue                              # that block DID execute -- not skipped
            issue -= (skip_to - skip_at - 1)          # instructions jumped over
            penalty += MEASURED_BRANCH_PENALTY        # the skip is itself a taken branch
            kind += " + skipped body"
    return {"issue": issue, "branch_penalty": penalty, "kind": kind}


def describe_registers(acadia):
    """Work out where each sequencer register gets its value from.

    A register is loaded over the bus, which the compiled program spells as an
    address write followed by the data landing in the register::

        001D0000 -> BUS_ADDR  |  REG0 -> NONE      <- cache base + word index
        BUS_DATA -> REG0      |  REG0 -> NONE

    A DSP unit's input loads the same way (``BUS_DATA -> DSP_ABn``); its output
    ``DSPn`` then drives a command length, so it is recorded under that ``DSPn``
    name to match the length symbol the schedule carries.

    So walking back from each ``BUS_DATA -> REGn`` / ``DSP_ABn`` to the preceding
    ``-> BUS_ADDR`` identifies the source. An address inside the cache region gives
    the exact word -- which the dry run captures per sweep point, so the register's
    real value is known. Any other device (a CMACC accumulator, i.e. a measurement
    result) is named but has no static value.

    :return: ``{"REG0": {"source": "cache[0]", "cache_word": 0}, ...}``
    """
    prog = decode_program(acadia)

    decoder = acadia._firmware.sequencer_bus_decoder
    cache_base = decoder["cache"].address().value()
    cache_words = acadia._firmware["sequencer_cache_memory"]["size_bits"] // 8

    devices = {}
    for name in getattr(decoder, "keys", lambda: [])():
        try:
            devices[decoder[name].address().value()] = name
        except Exception:
            pass

    registers = {}
    for r in prog:
        loaded = _bus_data_load(r, ("REG", "DSP_AB"))   # BUS_DATA -> REGn / DSP_ABn
        if loaded is None:
            continue
        # Walk back to the bus address THIS read used. The walk must stop at the nearest
        # BUS_ADDR write of EITHER kind: a literal, or a pointer (BUS_ADDR <- DSP_P, which is
        # what bus_read(pointer) emits). Only literals used to count, so a pointer-driven read
        # was walked straight past and attributed to whatever literal came before it -- in
        # DualRail2XEBRuntime that was the neighbouring fifo poll, and the viewer confidently
        # labelled the streamed gate commands "REG0 = dac3_dma", a channel they have nothing to
        # do with. A wrong label is worse than no label; a pointer read is now named as one.
        address, pointer = None, False
        for j in range(r.i - 1, max(r.i - 12, -1), -1):
            if _bus_addr_pointer(prog[j]):
                pointer = True
                break
            found = _bus_addr(prog[j])
            if found is not None:
                address = found
                break
        if pointer:
            major, minor = loaded
            name = f"DSP{minor}" if major == "DSP_AB" else f"REG{minor}"
            registers[name] = {"source": "cache[pointer]", "cache_word": None}
            continue
        if address is None:
            continue
        # a DSP unit's output DSPn drives the command length, not DSP_ABn (its input)
        major, minor = loaded
        name = f"DSP{minor}" if major == "DSP_AB" else f"REG{minor}"
        if cache_base <= address < cache_base + cache_words:
            word = address - cache_base
            registers[name] = {"source": f"cache[{word}]", "cache_word": word}
        else:
            registers[name] = {
                "source": devices.get(address, f"bus 0x{address:X}"),
                "cache_word": None}
    return registers


def describe_immediates(acadia):
    """``{"REG2": 1900548, "DSP0": 1900545}`` -- the compile-time constant each register or DSP
    counter was INITIALISED with.

    Separate from :func:`describe_registers`, which answers a different question: where a register
    gets its value at RUN time (a cache word, a bus device, a pointer read). A constant load is
    not a source in that sense -- it is a number the program carries -- and conflating the two
    would make a pointer's base look like a data source.

    Only the FIRST load of each name is kept. A counter is initialised once and then incremented
    by ``pulse_cep()``; a later immediate write to the same register is a different value's life,
    and taking it would report the end of a reused register as the start of a loop.
    """
    immediates = {}
    for r in decode_program(acadia):
        for dest, minor, imm, src in ((r.d1, r.d1_minor, r.imm1, r.s1),
                                      (r.d2, r.d2_minor, r.imm2, r.s2)):
            if src != "IMM" or dest not in ("REG", "DSP_AB"):
                continue
            # DSP_ABn is the DSP's INPUT; the counter is read back as DSPn, which is the name
            # every condition string uses -- so record it under the name the condition will ask
            # for, not the one the instruction writes.
            name = f"DSP{minor}" if dest == "DSP_AB" else f"REG{minor}"
            value = _resolve_imm(imm)
            if value is not None and name not in immediates:
                immediates[name] = value
    return immediates


def _dsp_pointer_config(r):
    """The DSP index a ``P+1`` config is written to (the walking pointer), else None.
    ``P+1`` means "from now on only increment", so this marks the cache-stream pointer DSP."""
    for dest, minor, imm in ((r.d1, r.d1_minor, r.imm1), (r.d2, r.d2_minor, r.imm2)):
        if dest == "DSP_CFG":
            mode = getattr(imm, "mode", None)
            if mode == "P+1" or getattr(mode, "name", None) == "P+1":
                return minor
    return None


def direct_command_words(acadia):
    """``{channel: cache word}`` for direct DMA commands read from a FIXED cache address.

    :func:`describe_cache_stream` handles the WALKING-pointer idiom (randomized benchmarking):
    a DSP steps through a region of cached DMA words, one gate per pass. Not every streamed
    sequence is built that way. ``BeamsplitterAmpDetuneCalibrationRuntime`` loads a plain
    ``Register`` with ONE cache address and replays that single word inside a deterministic
    ``loop(4N)`` -- no ``P+1`` DSP, no count word -- so the stream detector bails out and every
    play is left symbolic with ``resolve_indeterminate`` (0 cycles) for its length.

    Zero is badly wrong, and not only for the gate itself: 64 commands of length 0 collapse the
    whole train, so the readout block after it is drawn ~1.2 us EARLY and the experiment's
    structure disappears. The word is right there in the captured cache -- ``cache[0] = 0x19``
    decodes as an ARB of 26 cycles (130 ns) -- so the length is recoverable exactly.

    Nothing here needs the loop: the block containing the command is already unrolled once per
    iteration by :meth:`SequenceTrace.execution_plan`. Only the LENGTH is missing.

    Resolves the read address through either form acadia emits for a constant:
    ``BUS_ADDR <- IMM`` directly, or ``BUS_ADDR <- REG`` where that register was loaded from an
    immediate. A pointer that is stepped (``DSP_P``, or a register written more than once) is
    deliberately NOT resolved here -- its value differs per play, which is the walking-pointer
    case :func:`describe_cache_stream` owns.
    """
    prog = decode_program(acadia)
    decoder = acadia._firmware.sequencer_bus_decoder
    cache_base = decoder["cache"].address().value()
    cache_words = acadia._firmware["sequencer_cache_memory"]["size_bits"] // 8

    def in_cache(value):
        return value is not None and cache_base <= value < cache_base + cache_words

    # registers loaded from an immediate, and how many times -- a register written more than
    # once is being stepped, so its value is not a constant we may rely on
    reg_value, reg_writes = {}, {}
    for r in prog:
        for dest, minor, src, imm in ((r.d1, r.d1_minor, r.s1, r.imm1),
                                      (r.d2, r.d2_minor, r.s2, r.imm2)):
            if dest != "REG":
                continue
            reg_writes[minor] = reg_writes.get(minor, 0) + 1
            reg_value[minor] = _resolve_imm(imm) if src == "IMM" else None

    constant_pointers = sorted({v for m, v in reg_value.items()
                                if reg_writes.get(m) == 1 and in_cache(v)})

    words = {}
    for r in prog:
        if not (r.comment and r.comment.startswith("Command DMA for")):
            continue
        if not ((r.d1 == "BUS_DATA" and r.s1 == "BUS_DATA")
                or (r.d2 == "BUS_DATA" and r.s2 == "BUS_DATA")):
            continue
        match = re.match(r"Command DMA for (\w+)", r.comment)
        if not match:
            continue
        address = None
        for j in range(r.i - 1, max(r.i - 12, -1), -1):
            prev = prog[j]
            if _bus_addr_pointer(prev):
                address = None                     # stepped pointer: not a constant
                break
            literal = _bus_addr(prev)
            if literal is not None:
                address = literal
                break
            from_reg = any(d == "BUS_ADDR" and s == "REG"
                           for d, s in ((prev.d1, prev.s1), (prev.d2, prev.s2)))
            if from_reg:
                # WHICH register drives BUS_ADDR is not in the decoded record -- the minor on
                # this instruction is the destination bus port, not the source register (the
                # same trap that made describe_registers name a wrong device). So resolve it
                # only when the answer is unambiguous: exactly one register in the whole
                # program was loaded, once, with a constant cache address. Anything else is
                # left unresolved rather than guessed.
                if len(constant_pointers) == 1:
                    address = constant_pointers[0]
                break
        if in_cache(address):
            words[match.group(1)] = address - cache_base
    return words


def describe_cache_stream(acadia):
    """Locate a cache-pointer pulse stream (the randomized-benchmarking idiom).

    A DSP holds a bus pointer into a command cache and walks it -- loaded once, then
    configured ``P+1`` and pulsed each iteration -- issuing each cached word straight to the
    DMA via ``schedule_direct`` until it reaches a final pointer. Off-hardware the played
    word is unknown (it comes through as ``BUS_DATA -> BUS_DATA`` with a "Command DMA" note),
    but the cache is captured per point, so the whole train is recoverable: the pointer's
    initial address gives where the command region starts, and the final pointer is
    ``start + cache[count word]``, so the loop runs ``cache[count word]`` times.

    :return: ``{"channel", "start_offset", "count_word", "floor"}`` (``channel_num`` is filled
        in by the caller from the runtime), or ``None`` when there is no such stream.
    """
    prog = decode_program(acadia)

    # the played command: BUS_DATA issued straight to a channel's DMA, named in the comment
    direct = None
    for r in prog:
        if (r.comment and r.comment.startswith("Command DMA for")
                and ((r.d1 == "BUS_DATA" and r.s1 == "BUS_DATA")
                     or (r.d2 == "BUS_DATA" and r.s2 == "BUS_DATA"))):
            match = re.match(r"Command DMA for (\w+)", r.comment)
            if match:
                direct = (r.i, match.group(1))
                break
    if direct is None:
        return None
    direct_idx, channel = direct

    decoder = acadia._firmware.sequencer_bus_decoder
    cache_base = decoder["cache"].address().value()
    cache_words = acadia._firmware["sequencer_cache_memory"]["size_bits"] // 8

    # The walking pointer: a DSP reconfigured P+1 ("from now on only increment") whose AB input
    # is loaded with an address INSIDE the command cache.
    #
    # Both halves of that matter. Taking merely the first P+1 config in the program is wrong on
    # every real runtime: DualRailRBRuntime configures four DSPs P+1 (the cooling-round and
    # gate-sequence counters come first) and the cache pointer is the FOURTH. Picking DSP 0 found
    # an AB immediate of 0, which is not a cache address, so the whole stream went undetected --
    # the gate train then stayed a single symbolic BUS_DATA command and drew as one grey block
    # instead of the individual gates. A synthetic case with one P+1 counter hides this
    # completely, which is why the harness's own rb_stream case passed throughout.
    #
    # Pointing into the cache is what makes a counter a cache pointer, so select on that.
    pointers = {c for r in prog if (c := _dsp_pointer_config(r)) is not None}
    cfg = start_offset = None
    for r in prog:
        for dest, minor, src, imm in ((r.d1, r.d1_minor, r.s1, r.imm1),
                                      (r.d2, r.d2_minor, r.s2, r.imm2)):
            if dest != "DSP_AB" or minor not in pointers or src != "IMM":
                continue
            addr = _resolve_imm(imm)
            if addr is not None and cache_base <= addr < cache_base + cache_words:
                cfg, start_offset = minor, addr - cache_base
                break
        if cfg is not None:
            break
    if cfg is None:
        return None

    # count_word: the cache read feeding the AB+C 'count' term of the final-pointer DSP
    count_word = None
    for r in prog:
        if _bus_data_load(r, ("DSP_C",)) is not None:
            for j in range(r.i - 1, max(r.i - 12, -1), -1):
                addr = _bus_addr(prog[j])
                if addr is not None:
                    if cache_base <= addr < cache_base + cache_words:
                        count_word = addr - cache_base
                    break
            break

    # floor: the per-pulse period floor = the loop body's instruction span + the taken-branch
    # penalty (measured 22 cyc = ~110 ns, the fifo-refill floor). Use the back-edge whose span
    # contains the direct command.
    #
    # post_span: the trailing the LAST gate pays before the block after the loop -- counted as
    # the instruction span from the gate's own trigger to that next block's trigger, NOT another
    # full floor. The floor is the push cadence *between* gates (each waits a loop-body span to
    # refill for the next); the last gate has no next gate to refill for, so it only pays the
    # loop exit + post-loop drain + the next block's push/trigger. Emergent from the program, so
    # a stream followed by a batch (which starts exactly at this trailing) lands right, where a
    # blanket count*floor put it ~one floor too late.
    triggers = [r.i for r in prog if _is_trigger(r)]
    gate_trigger = next((t for t in triggers if t > direct_idx), None)
    floor = post_span = None
    for branch, target in sequencer_control_flow(acadia)["back_branches"]:
        if target <= direct_idx <= branch:
            floor = (branch - target + 1) + MEASURED_BRANCH_PENALTY
            next_trigger = next((t for t in triggers if t > branch), None)
            if gate_trigger is not None and next_trigger is not None:
                post_span = next_trigger - gate_trigger
            break

    if start_offset is None or count_word is None or floor is None:
        return None
    return {"channel": channel, "start_offset": start_offset, "count_word": count_word,
            "floor": floor, "post_span": post_span if post_span is not None else floor}


INT16_FULL_SCALE = 2 ** 15 - 1


def _compute_config_envelopes(runtime, trace):
    """Nominal waveform per pulse, from the config.

    The per-point *loaded* waveforms come from the snapshots instead -- read
    straight out of DAC memory, which under ``fake_attach`` is a host buffer
    holding exactly what ``load_pulse`` wrote.
    """
    wanted = {(c.io_name, c.pulse) for c in trace.commands
              if c.io_name is not None and c.pulse is not None}
    for io_name, pulse in wanted:
        try:
            trace.envelopes[(io_name, pulse)] = runtime.io(io_name).compute_pulse(
                pulse, return_raw=False)
        except Exception:
            pass
