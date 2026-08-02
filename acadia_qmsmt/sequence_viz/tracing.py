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

from .dryrun import (StopDryRun, already_traced, branch_recorder,
                     hardware_stubbed, preserved_runtime_state)

KIND = {
    DMASynchronizer.ARBITRARY_CONTINUED: "ARB_CONT",
    DMASynchronizer.ARBITRARY: "ARB",
    DMASynchronizer.CONSTANT_CONTINUED: "CONST_CONT",
    DMASynchronizer.DWELL: "DWELL",
    DMASynchronizer.DIRECT: "DIRECT",
}

# Markers identifying the two instructions that bracket a blocking block's wait,
# as emitted by DMASynchronizer.__exit__ / Sequencer.repeat_until.
TRIGGER_MARKER = "; Trigger DMAs"
DMA_POLL_MARKER = "PC (absolute hold) if BUS_DATA AND MASK != 0"

# Measured, not derived. The three terms below (detect + issue + propagate) come out one
# cycle short of hardware at every blocking block boundary. Established on the 4-channel
# DAC->ADC loopback (validation/timing_validation.py, 2026-07-27): the error is
# +1.000 cycle for one boundary and +2.00 for two, and does not change with the number of
# DMA pushes (1 / 2 / 4 channels all give +1), while intra-block layout -- back-to-back
# pulses, dwell(), and block=False FIFO batching -- is exact to 0.01 cycle. So the shortfall
# is a fixed per-boundary constant, not a counting error. Which term owns it is not
# separable by timing alone: candidates are Acadia._bus_latency("dma_running") being one
# low, the trigger->DMA-load propagation being one low, or a DAC start latency that is not
# modelled at all. Kept as its own named term rather than silently folded into one of them.
MEASURED_BOUNDARY_OFFSET = 1

# Also measured. A boundary crossed by a TAKEN branch (a loop back-edge) costs three more
# cycles than the straight-line instruction count, on top of MEASURED_BOUNDARY_OFFSET --
# consistent with a pipeline flush on the redirected fetch. Established on the loopback with
# two different loop bodies (timing_validation.py cases loop_2 / loop_3 / loop_2_double,
# 2026-07-27): a 4-push body measured 20 cycles against 11 counted instructions, an 8-push
# body 21 against 12. The counting tracks the body exactly; only this constant is extra.
MEASURED_BRANCH_PENALTY = 3

CONDITION_RE = re.compile(r"^(REG\d+)\s*(==|!=|<=|>=|<|>)\s*(\S+)$")
BRANCH_MARKER = "-> PC (absolute branch)"
# Two spellings of a branch target appear in the compiled program, depending on how the
# target instruction was referenced: a loop back-edge prints "SequenceInstruction @ 00000059",
# a `test` skip prints "Symbol(assigned=True, value=0x7B)". Missing the second form silently
# cost the whole skip-edge gap (the blocks butted together instead).
BRANCH_TARGET_RE = re.compile(r"(?:@\s*|value=0x)([0-9A-Fa-f]+)")


@dataclass
class Command:
    """One DMA command placed on the timeline. Times are in sequencer cycles."""
    channel: str
    kind: str
    start: int
    length: int
    is_padding: bool = False
    symbolic: Optional[str] = None
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
    path_choices: dict = field(default_factory=dict)  # block index -> take test body?
    assumed_paths: set = field(default_factory=set)   # tests we could not decide
    unsupported_paths: set = field(default_factory=set)  # KI_004: speculation=False
    gap_terms: dict = field(default_factory=dict)   # detect/propagate, from firmware
    registers: dict = field(default_factory=dict)         # "REG0" -> {source, cache_word}
    register_sources: dict = field(default_factory=dict)  # "REG0" -> cache word
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
        """
        fallback = int(self.resolve_indeterminate)
        self.unresolved = 0
        self.placements = []
        self.assumed_paths = set()
        self.unsupported_paths = set()

        # trigger-list index per block, in compiled order: gaps are costed against the
        # compiled program, so an executed edge is (from nth trigger -> to nth trigger)
        nth_of_block, nth = {}, -1
        for index, block in enumerate(self.blocks):
            if block.trigger:
                nth += 1
                nth_of_block[index] = nth

        plan = self.execution_plan()

        # The sequencer is a single instruction stream: `t_seq` is when it reaches the
        # current block. A blocking block waits for its DMAs, so it advances t_seq for
        # everyone; a non-blocking one only queues into each channel's FIFO, so following
        # blocks resume from that channel's own cursor.
        cursor, t_seq = {}, 0
        for step, (index, iteration) in enumerate(plan):
            block = self.blocks[index]
            placement = Placement(index=index, iteration=iteration,
                                  trigger=block.trigger, blocking=block.blocking,
                                  conditional=block.conditional)
            channels = {c.channel for c in block.commands}
            placement.start = max([t_seq] + [cursor.get(c, 0) for c in channels])

            stream_here = False
            t_sub = placement.start
            for i_sub, group in enumerate(block.subschedules):
                per_channel, sub_len = {}, 0
                for command in group:
                    if self._is_stream_command(command):
                        # a cache-pointer pulse stream: unroll it into one concrete command
                        # per cached word (randomized benchmarking), spaced by the period
                        stream_here = True
                        start = per_channel.get(command.channel, t_sub)
                        end = self._expand_stream(command, placement, start)
                        per_channel[command.channel] = end
                        sub_len = max(sub_len, end - t_sub)
                        continue
                    length = command.length
                    resolution = command.resolution
                    if command.symbolic:
                        # an explicit override wins over the cache, which wins over the
                        # blanket resolve_indeterminate fallback
                        override = self.register_overrides.get(command.symbolic)
                        resolved = self.register_cycles.get(command.symbolic)
                        value = (override if override is not None else
                                 resolved if resolved is not None else fallback)
                        length = int(value)
                        resolution = ("override" if override is not None
                                      else "cache" if resolved is not None
                                      else "fallback")
                        self.unresolved += 1
                    start = per_channel.get(command.channel, t_sub)
                    placement.commands.append(replace(
                        command, start=start, length=length, resolution=resolution))
                    per_channel[command.channel] = start + length
                    sub_len = max(sub_len, per_channel[command.channel] - t_sub)
                t_sub += sub_len
                if i_sub < len(block.subschedules) - 1:
                    placement.barriers.append(t_sub)

            placement.length = t_sub - placement.start
            placement.stream = stream_here

            # gap to whatever executes NEXT, which for a loop back-edge is not the
            # textually-following block
            if placement.blocking and step + 1 < len(plan):
                from_nth = nth_of_block.get(index)
                to_nth = nth_of_block.get(plan[step + 1][0])
                if from_nth is not None and to_nth is not None:
                    edge = edge_gap(self.control_flow, from_nth, to_nth)
                    if edge:
                        detect = self.gap_terms.get("detect", 3)
                        propagate = self.gap_terms.get("propagate", 2)
                        total = (detect + edge["issue"] + propagate
                                 + MEASURED_BOUNDARY_OFFSET + edge["branch_penalty"])
                        placement.gap_after = total
                        placement.gap_breakdown = {
                            "total": total, "detect": detect, "issue": edge["issue"],
                            "propagate": propagate,
                            "measured_offset": MEASURED_BOUNDARY_OFFSET,
                            "branch_penalty": edge["branch_penalty"],
                            "edge": edge["kind"]}

            if placement.blocking or stream_here:
                # A cache-pointer stream is non-blocking per se, but the runtime waits for
                # its DAC FIFO to drain (the post-loop fifo-almost-empty poll) before the
                # next block, which holds the whole sequencer -- so, like a blocking block,
                # it advances t_seq for every channel, not just its own. Without this the
                # readout (on other channels) would be drawn on top of the gate train.
                t_seq = placement.stop + placement.gap_after
                for ch in channels:
                    cursor[ch] = t_seq
            else:
                for ch in channels:
                    own = [c.stop for c in placement.commands if c.channel == ch]
                    cursor[ch] = max(own, default=placement.stop)

            self.placements.append(placement)
        return self

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
            cursor += max(length, floor)      # advance by the per-pulse period
        # The loop runs `count` full period slots, so the train ends `count` periods in --
        # the last slot's refill gap is where the loop does its exit check and the post-loop
        # fifo drains before the next block. Ending at the last pulse's stop instead would
        # butt the next block against the final gate and merge them (which the hardware,
        # separated by that drain, does not). The residual (~one refill gap) is small.
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

    def execution_plan(self):
        """``[(block index, iteration), ...]`` in the order the sequencer runs them.

        Consecutive blocks sharing the same innermost ``loop``/``repeat_until`` context form
        one body, repeated ``loop_counts[first block]`` times (default: the loop's own count
        for ``loop``, 1 for ``repeat_until``, whose count is data-dependent).
        """
        plan, index = [], 0
        while index < len(self.blocks):
            block = self.blocks[index]
            context = block.conditional[-1] if block.conditional else None
            if context is None:
                plan.append((index, 0))
                index += 1
                continue

            body = [index]
            while (body[-1] + 1 < len(self.blocks)
                   and self.blocks[body[-1] + 1].conditional
                   and self.blocks[body[-1] + 1].conditional[-1] == context):
                body.append(body[-1] + 1)

            if context["kind"] == "test" and context.get("speculation") is False:
                # KI_004: with speculation=False the body is placed OUT OF LINE, so address
                # order stops being execution order and the edge costing below does not
                # apply -- measured 25 ns out on the taken arm, and the skipped arm hangs
                # the sequencer outright. Draw the body but flag the path as unmodelled
                # rather than assert a timeline we know is wrong.
                self.assumed_paths.add(index)
                self.unsupported_paths.add(index)
                plan.extend((member, 0) for member in body)
            elif context["kind"] == "test":
                # explicit choice wins; otherwise try to decide it from the cache;
                # otherwise assume the body runs (and say so via `assumed_paths`)
                taken = self.path_choices.get(index)
                if taken is None:
                    taken = self.evaluate_condition(context.get("condition"))
                    if taken is None:
                        taken = True
                        self.assumed_paths.add(index)
                if taken:
                    plan.extend((member, 0) for member in body)
            else:
                # loop: deterministic count, unrolled. repeat_until: count is
                # data-dependent, so one pass, labelled as such by the renderer.
                count = self.loop_counts.get(index, context.get("count") or 1)
                for iteration in range(max(int(count), 1)):
                    plan.extend((member, iteration) for member in body)
            index = body[-1] + 1
        return plan

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

    def register_summary(self):
        """Describe every register / length-symbol for a per-register UI control.

        Two kinds appear, distinguished by ``settable``:

        * cache-fed registers resolve themselves from the per-point cache, so
          their value is shown but not settable. One may drive a command length
          (``is_length``, shown in cycles/ns) or only a ``test``/``repeat_until``
          condition (shown as the raw register value).
        * register/DSP-driven command *lengths* not recoverable from the cache
          (``resolution`` "fallback"/"override") -- the only thing worth setting,
          via :attr:`register_overrides`.

        :return: ``[{name, label, source, resolution, value_cycles, is_length,
            settable}, ...]``, deduped by name, identified registers first.
        """
        resolved = {c.symbolic: (c.resolution, c.length)
                    for c in self.commands if c.symbolic}

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
        """Total inter-block dead time -- see :func:`sequencer_block_gaps`."""
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


def sequencer_block_gaps(acadia):
    """Dead time between one blocking block's last sample and the next block's first.

    Standalone diagnostic for the straight-line case, kept as the readable reference
    for the gap model (see README). The live layout no longer calls this -- it uses
    :func:`edge_gap` via :meth:`SequenceTrace.relayout`, which generalises the same
    count to loop/``test`` edges. Not wired into the trace; call it directly.

    A blocking ``channel_synchronizer`` does not hand off seamlessly. At block
    exit ``DMASynchronizer.__exit__`` emits, in order: the DMA trigger, a
    ``bus_read`` of ``dma_running``, and a ``repeat_until`` that holds the PC
    until the mask clears. Only once that poll releases does the sequencer push
    the next block's DMA commands, pad with the FIFO-latency NOPs from
    ``calculate_trigger_delay``, and trigger again. Every one of those cycles is
    dead air on every channel.

    Three contributions, keyed to where they come from:

    ``detect``
        ``Acadia._bus_latency("dma_running")`` -- the poll reads a value that
        stale, so the deassertion is seen that many cycles late.
    ``issue``
        counted exactly out of the compiled program: instructions from the poll
        instruction through the next ``Trigger DMAs``. This is where a block with
        many pulses, or one preceded by datamover configuration, costs more.
    ``propagate``
        trigger-to-DMA-load: ``dma_trigger_dataport`` pipelining plus the one
        cycle the DMA takes to latch the FIFO output (see the ``calculate_trigger_delay``
        docstring). Counted once, since it applies equally to both blocks.

    Non-blocking blocks get no entry: their commands queue in the DMA FIFO and
    play back-to-back, which is exactly why batching with ``block=False`` avoids
    this cost.

    :return: ``{nth triggering block: {"total", "detect", "issue", "propagate"}}``
    """
    text = [instruction.pprint()
            for sequencer in acadia._sequencer_type.instances
            for instruction in sequencer._compiled_program]

    triggers = [i for i, line in enumerate(text) if TRIGGER_MARKER in line]
    detect = acadia._bus_latency("dma_running")
    dataport = acadia._firmware["sequencer_bus"]["dma_trigger_dataport"]
    propagate = (max(dataport["pipeline"])
                 + (1 if dataport["bus_pipeline"] else 0)
                 + 1)

    gaps = {}
    for nth, trigger in enumerate(triggers[:-1]):
        following = triggers[nth + 1]
        poll = next((j for j in range(trigger + 1, following)
                     if DMA_POLL_MARKER in text[j]), None)
        if poll is None:
            continue                      # non-blocking: no wait, no gap
        issue = following - poll + 1       # poll's last pass ... trigger inclusive
        gaps[nth] = {"total": detect + issue + propagate + MEASURED_BOUNDARY_OFFSET,
                     "detect": detect, "issue": issue, "propagate": propagate,
                     "measured_offset": MEASURED_BOUNDARY_OFFSET}
    return gaps


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

    # A cache-pointer pulse stream (randomized benchmarking) is unrolled from the per-point
    # cache in relayout; store the decode map and the stream descriptor here.
    trace.addr_names = addr_names
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
                    symbolic = length = None
                    if isinstance(raw_len, (int, np.integer)):
                        length = int(raw_len) + (1 if c.get("length_is_minus_one")
                                                 else 0)
                    else:
                        symbolic = str(raw_len)
                    io_name = pulse = None
                    address = c.get("address")
                    if address is not None:
                        io_name, pulse = addr_names.get(
                            (ch.num(), address), (None, None))
                    group.append(Command(
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


LOAD_RE = re.compile(r"BUS_DATA -> (REG\d+|DSP_AB\d+)")
BUS_ADDR_RE = re.compile(r"(?:0x([0-9A-Fa-f]+)|\b([0-9A-F]{8})\b)\s*->\s*BUS_ADDR")


def sequencer_control_flow(acadia):
    """Trigger/poll addresses and backward branches, for costing an executed edge.

    Straight-line edges can be costed by counting instructions between the poll and the
    next trigger. A loop back-edge cannot: execution leaves the poll, reaches the branch,
    jumps to the branch target, and runs forward from there to that block's trigger. So the
    branch and its target address are needed as well.

    :return: ``{"triggers": [addr], "polls": {nth trigger: addr},
                "back_branches": [(branch addr, target addr)]}``
    """
    text = [instruction.pprint()
            for sequencer in acadia._sequencer_type.instances
            for instruction in sequencer._compiled_program]

    triggers = [i for i, line in enumerate(text) if TRIGGER_MARKER in line]
    polls = {}
    for nth, trigger in enumerate(triggers):
        limit = triggers[nth + 1] if nth + 1 < len(triggers) else len(text)
        poll = next((j for j in range(trigger + 1, limit)
                     if DMA_POLL_MARKER in text[j]), None)
        if poll is not None:
            polls[nth] = poll

    # Both directions matter: a loop back-edge jumps backward, while a `test` whose
    # condition fails jumps FORWARD past the skipped body. Either way the executed path
    # leaves the address order and has to be costed in two segments.
    branches = []
    for index, line in enumerate(text):
        # the instruction's own destination is the TAIL of the line; a Symbol repr earlier
        # in it quotes a different instruction and must not be mistaken for this one
        head, marker, _ = line.rpartition(BRANCH_MARKER)
        if not marker:
            continue
        targets = BRANCH_TARGET_RE.findall(head)
        if targets:
            branches.append((index, int(targets[-1], 16)))

    return {"triggers": triggers, "polls": polls, "branches": branches,
            "back_branches": [(b, t) for b, t in branches if t < b]}


def edge_gap(control_flow, from_nth, to_nth):
    """Dead cycles between two executed blocks, and how it breaks down.

    ``from_nth``/``to_nth`` index the compiled program's trigger list. A backward edge picks
    up the branch instructions and the taken-branch penalty.
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
    text = [instruction.pprint()
            for sequencer in acadia._sequencer_type.instances
            for instruction in sequencer._compiled_program]

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
    for index, line in enumerate(text):
        loaded = LOAD_RE.search(line)
        if not loaded:
            continue
        address = None
        for previous in range(index - 1, max(index - 12, -1), -1):
            found = BUS_ADDR_RE.search(text[previous])
            if found:
                address = int(found.group(1) or found.group(2), 16)
                break
        if address is None:
            continue
        # a DSP unit's output DSPn drives the command length, not DSP_ABn (its input)
        name = re.sub(r"^DSP_AB(\d+)$", r"DSP\1", loaded.group(1))
        if cache_base <= address < cache_base + cache_words:
            word = address - cache_base
            registers[name] = {"source": f"cache[{word}]", "cache_word": word}
        else:
            registers[name] = {
                "source": devices.get(address, f"bus 0x{address:X}"),
                "cache_word": None}
    return registers


# The randomized-benchmarking pulse stream, spelled in the compiled program:
#   <addr> -> DSP_AB0                                   pointer's initial cache address
#   mode='P+1' -> DSP_CFG0                              from now on it only increments
#   BUS_DATA -> DSP_C1                                  final = pointer + cache[count word]
#   BUS_DATA -> BUS_DATA ; Command DMA for <chan>       the played word, fetched at runtime
STREAM_DIRECT_RE = re.compile(r"BUS_DATA -> BUS_DATA\b.*Command DMA for (\w+)")
STREAM_PTR_CFG_RE = re.compile(r"mode='P\+1'.*->\s*DSP_CFG(\d+)")
STREAM_C_RE = re.compile(r"BUS_DATA -> DSP_C\d+")


def describe_cache_stream(acadia):
    """Locate a cache-pointer pulse stream (the randomized-benchmarking idiom).

    A DSP holds a bus pointer into a command cache and walks it -- loaded once, then
    configured ``P+1`` and pulsed each iteration -- issuing each cached word straight to the
    DMA via ``schedule_direct`` until it reaches a final pointer. Off-hardware the played
    word is unknown (it shows up as ``BUS_DATA -> BUS_DATA``), but the cache is captured per
    point, so the whole train is recoverable: the pointer's initial address gives where the
    command region starts, and the final pointer is ``start + cache[count word]``, so the
    loop runs ``cache[count word]`` times.

    :return: ``{"channel", "start_offset", "count_word", "floor"}`` (``channel_num`` is filled
        in by the caller from the runtime), or ``None`` when there is no such stream.
    """
    text = [instruction.pprint()
            for sequencer in acadia._sequencer_type.instances
            for instruction in sequencer._compiled_program]

    direct = next(((i, m.group(1)) for i, line in enumerate(text)
                   if (m := STREAM_DIRECT_RE.search(line))), None)
    if direct is None:
        return None
    direct_idx, channel = direct

    # the walking pointer: the DSP reconfigured P+1 (loaded once, then only incremented)
    cfg = next((m.group(1) for line in text
                if (m := STREAM_PTR_CFG_RE.search(line))), None)
    if cfg is None:
        return None

    decoder = acadia._firmware.sequencer_bus_decoder
    cache_base = decoder["cache"].address().value()
    cache_words = acadia._firmware["sequencer_cache_memory"]["size_bits"] // 8

    # start_offset: the immediate loaded into that DSP's AB input (its first cache address)
    ab_re = re.compile(rf"\b([0-9A-Fa-f]{{8}}) -> DSP_AB{cfg}\b")
    start_offset = None
    for line in text:
        found = ab_re.search(line)
        if found:
            addr = int(found.group(1), 16)
            if cache_base <= addr < cache_base + cache_words:
                start_offset = addr - cache_base
            break

    # count_word: the cache read feeding the AB+C 'count' term of the final-pointer DSP
    count_word = None
    for i, line in enumerate(text):
        if STREAM_C_RE.search(line):
            for j in range(i - 1, max(i - 12, -1), -1):
                found = BUS_ADDR_RE.search(text[j])
                if found:
                    addr = int(found.group(1) or found.group(2), 16)
                    if cache_base <= addr < cache_base + cache_words:
                        count_word = addr - cache_base
                    break
            break

    # floor: the per-pulse period floor = the loop body's instruction span + the taken-branch
    # penalty (measured 22 cyc = ~110 ns, the fifo-refill floor). Use the back-edge whose span
    # contains the direct command.
    floor = None
    for branch, target in sequencer_control_flow(acadia)["back_branches"]:
        if target <= direct_idx <= branch:
            floor = (branch - target + 1) + MEASURED_BRANCH_PENALTY
            break

    if start_offset is None or count_word is None or floor is None:
        return None
    return {"channel": channel, "start_offset": start_offset,
            "count_word": count_word, "floor": floor}


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
