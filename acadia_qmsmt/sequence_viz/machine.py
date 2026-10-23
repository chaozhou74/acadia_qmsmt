"""Execution-model layout for the sequence tracer.

This is what :meth:`SequenceTrace.relayout` calls to place every DMA command on the
timeline. Rather than deriving *when* things play from the block structure, it runs the
compiled program through a per-channel DMA-FIFO two-clock model, so the DAC-output timeline
emerges from executing it. Two facts, read off acadia's compiled program rather than
guessed, are what let it resolve the ``block=False`` desynchronisation a single block-wide
start cannot:

* **Each channel plays its own DMA FIFO independently** from a single trigger, so a block's
  playout on channel ``c`` starts at ``max(t_seq, cursor[c])`` -- a *per channel* cursor,
  not one block-wide start. When the channels are lock-step (every all-blocking sequence)
  this collapses to a shared start; it only diverges once a batch has desynchronised them
  (e.g. a re-sync dwell that plays concurrently with the last pulse, not additively).

* **A ``block=False`` batch is drained by a ``repeat_until(fifo_empty)``**, and (per
  ``acadia_dma.vhd``) ``fifo_empty`` asserts when the *last descriptor is pulled* -- while it
  is still playing out -- because the playing descriptor has already left the FIFO. So after
  such a batch the sequencer resumes at *last-pulled*, not at the end of playout. These drain
  blocks carry an inverted poll (``condition_invert``) in the compiled program, which is how
  :func:`drain_block_issue` finds them; a ``repeat_until(pass)`` drain emits no synchronizer
  block, so the block structure alone cannot see it.

* **A poll is a PC hold on a DMA status bit**, so the sequencer moves on a fixed number of
  cycles after the status changes, or one cycle after reaching the hold if it had already
  changed (:func:`~.tracing.gateware_timing`). Blocking boundaries and drains share that rule.

The intra-block subschedule/barrier layout, the executed-path instruction count
(:func:`~.tracing.edge_gap`), the cache-stream unroll, and loop unrolling via
``execution_plan`` are shared with the rest of the tracer. Hardware-validated on the
4-channel loopback (see ``validation/``).
"""
from dataclasses import replace

import re

from .tracing import (DEFAULT_DMA_TIMING, DMA_START, HOLD_EXIT_CYCLES, Destination, Placement,
                      REGISTER_NAME_RE, _bus_addr, decode_program, edge_gap)


def _after_hold(x_hold, x_seen):
    """Cycle the instruction after a PC hold executes, and whether the hold waited.

    ``x_hold`` is the cycle the hold first executes, ``x_seen`` the first cycle its condition
    reads clear (both sequencer cycles)."""
    if x_seen > x_hold:
        return x_seen + HOLD_EXIT_CYCLES, True
    return x_hold + 1, False


def _is_hold(r):
    """A conditional PC-hold poll (``PC (absolute hold) if BUS_DATA AND MASK``)."""
    return (r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_HOLD
            and r.conditional and r.s2 == "BUS_DATA" and r.op == "and")


#: ``MASK`` of a ``channel_is_fifo_almost_empty`` poll (status bit 3); ``fifo_empty`` is ``0x2``.
ALMOST_EMPTY_MASK = 0x8


def drain_block_issue(acadia):
    """``{nth: {"polls": [(index, channel, almost_empty), ...], "next": index}}`` for each
    ``block=False`` batch drained by ``repeat_until(fifo_empty/almost_empty)``, keyed by
    trigger order (== block order).

    The i-th ``Trigger DMAs`` is block i. A block is a FIFO *drain* block when the poll
    following its trigger is an *inverted* hold: ``condition_invert`` True is the
    ``fifo_empty``/``almost_empty`` sense, versus the ``dma_running`` (blocking) poll
    which is un-inverted. These are the only blocks whose ``t_seq`` advances to
    last-pulled rather than to end-of-playout.

    The span from a poll through the next block's trigger (inclusive) is the sequencer
    work between that FIFO releasing and the next block playing, costed by the same hold
    rule as a blocking edge (see :func:`~.tracing.edge_gap`).

    Each poll reads ONE channel's DMA status (its ``BUS_ADDR`` names ``dacN_dma``), and the
    ``MASK`` before it says which bit: ``fifo_empty`` releases when that channel's last
    descriptor is pulled, ``fifo_almost_empty`` one descriptor earlier (one word left in the
    FIFO). A drain often polls a channel other than the ones its block plays on (a
    multi-channel streamed cycle may wait on one line only), and several polls in a row
    (the drains after a streamed loop) all have to release, so every consecutive poll is
    kept.
    """
    prog = decode_program(acadia)
    decoder = acadia._firmware.sequencer_bus_decoder
    dma_of_address = {}
    for name in decoder.keys():
        match = re.fullmatch(r"(dac|adc)(\d+)_dma", name)
        if match:
            dma_of_address[decoder[name].address().value()] = \
                f"{match.group(1).upper()}{match.group(2)}"
    triggers = [r.i for r in prog if r.comment == "Trigger DMAs"]
    drains = {}
    for nth, trigger in enumerate(triggers):
        limit = triggers[nth + 1] if nth + 1 < len(triggers) else len(prog)
        polls = []
        for j in range(trigger + 1, limit):
            if not _is_hold(prog[j]):
                continue
            if not prog[j].condition_invert:
                break                       # a blocking dma_running poll: not a drain
            channel, mask = None, None
            for k in range(j - 1, max(j - 8, trigger), -1):
                r = prog[k]
                if mask is None and r.d1 == "MASK":
                    mask = r.imm1
                if channel is None:
                    address = _bus_addr(r)
                    if address is not None:
                        channel = dma_of_address.get(address)
            if channel is None and polls:
                break                       # not a FIFO poll (a CMACC wait, say)
            polls.append((j, channel, mask == ALMOST_EMPTY_MASK))
        if polls:
            drains[nth] = {"polls": polls, "next": limit if nth + 1 < len(triggers) else None}
    return drains


def _direct_gate(trace, command, block):
    """``{length, address, io_name, pulse}`` of a ``schedule_direct`` command whose DMA word
    sits in a register loaded from the cache (``REG0 = cache[pointer]``),
    decoded as ``waveform_dma_command`` packs it; None when the word is not known here."""
    word = trace._resolve_leaf(command.symbolic, block.register_exprs)
    if word is None:
        return None
    word = int(word) & 0xFFFFFFFF
    address, length = word >> 16, (word & 0xFFFF) + 1
    io_name, pulse = trace.addr_names.get(
        (trace.channel_nums.get(command.channel), address), (None, None))
    return {"length": length, "address": address, "io_name": io_name, "pulse": pulse}


def machine_layout(trace):
    """Lay out ``trace.placements`` from the execution model. Called by
    :meth:`SequenceTrace.relayout`; mutates and returns ``trace``.

    For lock-step (all-blocking) sequences every channel shares one start; the per-channel
    cursors only diverge once a ``block=False`` batch has desynchronised the channels.
    """
    self = trace
    fallback = int(self.resolve_indeterminate)
    self.unresolved = 0
    self.placements = []
    self.assumed_paths = set()
    self.unsupported_paths = set()

    nth_of_block, nth = {}, -1
    for index, block in enumerate(self.blocks):
        if block.trigger:
            nth += 1
            nth_of_block[index] = nth

    plan = self.execution_plan()
    drains = self.drain_blocks or {}
    timing = {**DEFAULT_DMA_TIMING, **(self.dma_timing or {})}
    triggers = self.control_flow.get("triggers", [])
    pushes = self.control_flow.get("pushes", {})

    # Two clocks, both in cycles on the DMA output timeline: `t_seq` is where the sequencer's
    # program counter is, as the output time of a trigger it issued now (so the instruction
    # executes at `t_seq - timing["trigger"]`); `cursor[c]` is where channel c has played to. A blocking block resynchronises them for every
    # channel; a non-blocking batch lets `t_seq` run ahead of the cursors.
    cursor, t_seq = {}, 0
    # start times of the last two descriptors queued on each channel: a fifo_empty poll
    # releases when the last one starts playing, fifo_almost_empty when the one before does
    queued_starts = {}

    def next_executed_nth(step):
        """Trigger index of the next executed block that has a trigger of its own."""
        for ahead in range(step + 1, len(plan)):
            to_nth = nth_of_block.get(plan[ahead][0])
            if to_nth is not None:
                return to_nth
        return None

    for step, (index, iteration) in enumerate(plan):
        block = self.blocks[index]
        # where the cache pointers are on this pass, for a schedule_direct word
        self._pointer_extra = self.plan_pointers[step] if step < len(self.plan_pointers) else {}
        placement = Placement(index=index, iteration=iteration,
                              trigger=block.trigger, blocking=block.blocking,
                              conditional=block.conditional)
        channels = {c.channel for c in block.commands}

        # Per-channel start: each channel plays its own FIFO from the trigger, so it
        # begins at whichever is later -- the sequencer reaching the block (t_seq) or
        # that channel finishing what it was already playing (cursor[c]). Lock-step
        # channels share one value -- a single block-wide start. A channel still playing
        # when its first descriptor of this block lands pulls it on its last cycle and
        # plays on without waiting for the trigger.
        nth = nth_of_block.get(index)
        x_trigger = t_seq - timing["trigger"]
        play_start = {}
        for ch in channels:
            at = cursor.get(ch, 0)
            push = pushes.get(nth, {}).get(ch)
            seamless = (push is not None and at < t_seq
                        and (x_trigger - (triggers[nth] - push) + timing["fifo_ready"]
                             <= at - DMA_START))
            play_start[ch] = at if seamless else max(t_seq, at)
        stream_here = False
        # last descriptor length per channel, for the last-pulled drain time
        last_len = {}
        # each channel's block-relative end. For a cache stream this is the full period
        # cursor (the trailing fifo-refill period past the last gate), not just the last
        # gate's stop -- that trailing period is where the loop exit/drain happens and the
        # next block must sit after it, which the block length carries.
        chan_end_rel = {}

        # intra-block layout is channel-relative: barriers and padding fix each
        # command's offset from the block start independently of the absolute start,
        # so the same relative structure shifts per channel by its own play_start.
        t_sub = 0
        for i_sub, group in enumerate(block.subschedules):
            per_channel, sub_len = {}, 0
            for command in group:
                if self._is_stream_command(command):
                    stream_here = True
                    start_rel = per_channel.get(command.channel, t_sub)
                    base = play_start[command.channel]
                    end = self._expand_stream(command, placement, base + start_rel)
                    end_rel = end - base
                    per_channel[command.channel] = end_rel
                    last_len[command.channel] = 0
                    sub_len = max(sub_len, end_rel - t_sub)
                    continue
                length = command.length
                resolution = command.resolution
                gate = _direct_gate(self, command, block) if command.direct else None
                if gate is not None:
                    start_rel = per_channel.get(command.channel, t_sub)
                    placement.commands.append(replace(
                        command, start=play_start[command.channel] + start_rel,
                        resolution="cache", symbolic=None, **gate))
                    per_channel[command.channel] = start_rel + gate["length"]
                    last_len[command.channel] = gate["length"]
                    sub_len = max(sub_len, per_channel[command.channel] - t_sub)
                    continue
                if command.symbolic:
                    override = self.register_overrides.get(command.symbolic)
                    # what the register was loaded with at this block (a cache word, a
                    # pointer read, arithmetic on them), else the cache-fed map
                    resolved = (self._resolve_leaf(command.symbolic, block.register_exprs)
                                if REGISTER_NAME_RE.fullmatch(command.symbolic) else None)
                    if resolved is None:
                        resolved = self.register_cycles.get(command.symbolic)
                    else:
                        resolved = int(resolved) + command.length_offset
                    value = (override if override is not None else
                             resolved if resolved is not None else fallback)
                    length = int(value)
                    resolution = ("override" if override is not None
                                  else "cache" if resolved is not None
                                  else "fallback")
                    if command.static_length is None:     # not acadia's padding arithmetic
                        self.unresolved += 1
                start_rel = per_channel.get(command.channel, t_sub)
                placement.commands.append(replace(
                    command, start=play_start[command.channel] + start_rel,
                    length=length, resolution=resolution))
                per_channel[command.channel] = start_rel + length
                last_len[command.channel] = length
                sub_len = max(sub_len, per_channel[command.channel] - t_sub)
            for ch, end_rel in per_channel.items():
                chan_end_rel[ch] = end_rel     # this channel's latest block-rel end
            t_sub += sub_len
            if i_sub < len(block.subschedules) - 1:
                # barrier: absolute time differs per channel (each shifted by its own
                # play_start), so record it against the earliest channel for rendering
                placement.barriers.append(min(play_start.values(), default=0) + t_sub)

        # the block spans from its earliest channel start to its latest channel end
        # (each channel shifted by its own play_start), and each channel's cursor carries
        # to the next block from that end.
        chan_end = {ch: play_start[ch] + chan_end_rel.get(ch, 0) for ch in channels}
        placement.start = min(play_start.values(), default=t_seq)
        placement.length = (max(chan_end.values()) - placement.start) if chan_end else 0
        placement.stream = stream_here
        for ch in channels:
            cursor[ch] = chan_end.get(ch, play_start[ch])
            starts = sorted(c.start for c in placement.commands
                            if c.channel == ch and c.length)
            queued_starts[ch] = (queued_starts.get(ch, []) + starts)[-2:]

        # gap to whatever executes NEXT (loop back-edges included)
        if placement.blocking and step + 1 < len(plan):
            from_nth = nth_of_block.get(index)
            to_nth = next_executed_nth(step)
            if from_nth is not None and to_nth is not None:
                edge = edge_gap(self.control_flow, from_nth, to_nth)
                if edge:
                    # the dma_running hold: reached straight on from the trigger, released once
                    # it reads the playout's end, then the path to the next trigger
                    poll = self.control_flow["polls"][from_nth]
                    x_hold = x_trigger + poll - triggers[from_nth]
                    x_next, waited = _after_hold(x_hold,
                                                 placement.stop + timing["running_seen"])
                    t_next = (x_next + edge["issue"] - 2 + edge["branch_penalty"]
                              + timing["trigger"])
                    placement.gap_after = t_next - placement.stop
                    placement.gap_breakdown = {
                        "total": placement.gap_after, "issue": edge["issue"],
                        "branch_penalty": edge["branch_penalty"], "waited": waited,
                        "edge": edge["kind"]}

        # advance the sequencer clock to the next block
        if nth in drains and not stream_here:
            # a block=False drain releases when the polled channel's last descriptor is
            # pulled (fifo_empty) or the one before it (fifo_almost_empty) -- while it still
            # plays. Each poll is a hold of its own, in program order; after the last one the
            # sequencer issues the next block's pushes and trigger.
            # Channels nobody polls keep playing and carry over through their cursors.
            drain = drains[nth]
            j0 = drain["polls"][0][0]
            # the issue span along the EXECUTED path (a loop's back-edge, skipped tests),
            # not address order (address order is one cycle per pass off in a loop)
            to_nth = next_executed_nth(step)
            edge = (edge_gap(self.control_flow, nth, to_nth, start=j0)
                    if to_nth is not None else None)
            if edge:
                issue0, penalty = edge["issue"], edge["branch_penalty"]
            else:
                nxt = drain["next"]
                issue0, penalty = ((nxt - j0 + 1) if nxt is not None else 0), 0
            x_next, j_prev = x_trigger + j0 - triggers[nth], j0
            for j, ch, almost in drain["polls"]:
                x_hold = x_next + j - j_prev - (1 if j != j0 else 0)
                starts = queued_starts.get(ch, [])
                if ch is None:              # unknown channel: the block's own last pull
                    ref = max(cursor[c] - last_len.get(c, 0) for c in channels)
                elif len(starts) < (2 if almost else 1):
                    ref = None              # the FIFO never held more than that
                else:
                    ref = starts[-2] if almost else starts[-1]
                x_seen = x_hold if ref is None else ref + timing["fifo_seen"]
                x_next, _ = _after_hold(x_hold, x_seen)
                j_prev = j
            t_seq = x_next + issue0 - (j_prev - j0) - 2 + penalty + timing["trigger"]
        elif placement.blocking or stream_here:
            # blocking block (or a cache stream whose post-loop drain holds the whole
            # sequencer): playout completes for every channel before the next block.
            t_seq = placement.stop + placement.gap_after
            for ch in channels:
                cursor[ch] = t_seq
        else:
            # a plain non-blocking batch: channels keep playing, but the sequencer still runs
            # everything up to the next trigger first
            to_nth = next_executed_nth(step)
            if nth is not None and to_nth is not None and to_nth > nth:
                edge = edge_gap(self.control_flow, nth, to_nth,
                                start=self.control_flow["triggers"][nth] + 1)
                if edge:
                    t_seq += edge["issue"] + edge["branch_penalty"]

        self.placements.append(placement)
    return self
