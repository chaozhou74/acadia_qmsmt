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

The intra-block subschedule/barrier layout, the blocking-boundary gap model
(:func:`~.tracing.edge_gap` and the loopback-pinned constants), the cache-stream unroll, and
loop unrolling via ``execution_plan`` are shared with the rest of the tracer. Hardware-
validated on the 4-channel loopback (see ``validation/``).
"""
from dataclasses import replace

from .tracing import (Destination, Placement, decode_program, edge_gap,
                      MEASURED_BOUNDARY_OFFSET)


def _is_hold(r):
    """A conditional PC-hold poll (``PC (absolute hold) if BUS_DATA AND MASK``)."""
    return (r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_HOLD
            and r.conditional and r.s2 == "BUS_DATA" and r.op == "and")


def drain_block_issue(acadia):
    """``{nth: issue_cycles}`` for each ``block=False`` batch drained by a
    ``repeat_until(fifo_empty/almost_empty)``, keyed by trigger order (== block order).

    The i-th ``Trigger DMAs`` is block i. A block is a FIFO *drain* block when the poll
    following its trigger is an *inverted* hold: ``condition_invert`` True is the
    ``fifo_empty``/``almost_empty`` sense, versus the ``dma_running`` (blocking) poll
    which is un-inverted. These are the only blocks whose ``t_seq`` advances to
    last-pulled rather than to end-of-playout.

    ``issue_cycles`` is the instruction span from the drain poll through the next
    block's trigger (inclusive) -- the sequencer work between the FIFO emptying and the
    next block playing. With the firmware detect/propagate terms it forms the same
    boundary gap a blocking edge pays (see :func:`~.tracing.edge_gap`): the next block
    still starts one boundary gap after the drain releases, not immediately.
    """
    prog = decode_program(acadia)
    triggers = [r.i for r in prog if r.comment == "Trigger DMAs"]
    drains = {}
    for nth, trigger in enumerate(triggers):
        limit = triggers[nth + 1] if nth + 1 < len(triggers) else len(prog)
        for j in range(trigger + 1, limit):
            if _is_hold(prog[j]):
                if prog[j].condition_invert:
                    nxt = triggers[nth + 1] if nth + 1 < len(triggers) else None
                    drains[nth] = (nxt - j + 1) if nxt is not None else 0
                break
    return drains


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

    # Two clocks: `t_seq` is where the sequencer's program counter is; `cursor[c]` is
    # where channel c's DAC output has played to. A blocking block resynchronises them
    # for every channel; a non-blocking batch lets `t_seq` run ahead of the cursors.
    cursor, t_seq = {}, 0
    for step, (index, iteration) in enumerate(plan):
        block = self.blocks[index]
        placement = Placement(index=index, iteration=iteration,
                              trigger=block.trigger, blocking=block.blocking,
                              conditional=block.conditional)
        channels = {c.channel for c in block.commands}

        # Per-channel start: each channel plays its own FIFO from the trigger, so it
        # begins at whichever is later -- the sequencer reaching the block (t_seq) or
        # that channel finishing what it was already playing (cursor[c]). Lock-step
        # channels share one value -- a single block-wide start.
        play_start = {ch: max(t_seq, cursor.get(ch, 0)) for ch in channels}
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
                if command.symbolic:
                    override = self.register_overrides.get(command.symbolic)
                    resolved = self.register_cycles.get(command.symbolic)
                    value = (override if override is not None else
                             resolved if resolved is not None else fallback)
                    length = int(value)
                    resolution = ("override" if override is not None
                                  else "cache" if resolved is not None
                                  else "fallback")
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

        # gap to whatever executes NEXT (loop back-edges included)
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

        # advance the sequencer clock to the next block
        nth = nth_of_block.get(index)
        if nth in drains and not stream_here:
            # a block=False drain releases when the LAST descriptor is pulled -- the end
            # of playout minus that last (still-playing) descriptor. The sequencer then
            # still spends one boundary gap issuing the next block's pushes+trigger, so
            # the next block plays at last-pulled + that gap, not immediately.
            detect = self.gap_terms.get("detect", 3)
            propagate = self.gap_terms.get("propagate", 2)
            gap = detect + drains[nth] + propagate + MEASURED_BOUNDARY_OFFSET
            pulled = [cursor[ch] - last_len.get(ch, 0) for ch in channels]
            t_seq = max([t_seq] + pulled) + gap
        elif placement.blocking or stream_here:
            # blocking block (or a cache stream whose post-loop drain holds the whole
            # sequencer): playout completes for every channel before the next block.
            t_seq = placement.stop + placement.gap_after
            for ch in channels:
                cursor[ch] = t_seq
        # else: a plain non-blocking batch -- t_seq unchanged, channels keep playing.

        self.placements.append(placement)
    return self
