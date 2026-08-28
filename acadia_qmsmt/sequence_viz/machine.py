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
import re
from dataclasses import replace

from .tracing import (Destination, Placement, decode_program, edge_gap,
                      DMA_STATUS_REGISTER)


#: ``MASK`` value identifying a ``fifo_almost_empty`` poll, which releases one descriptor
#: earlier than ``fifo_empty`` (``0x2``). See :func:`drain_block_issue`.
ALMOST_EMPTY_MASK = 0x8



#: A `repeat_until` whose exit condition names a DSP -- the loop counter. When that counter is
#: also the cache POINTER the body reads its per-pass value through, the two facts together turn
#: an "indeterminate (register)" length into the actual number for every pass.
_LOOP_COUNTER_RE = re.compile(r"\b(DSP\d+)\b")


def _pointer_length(trace, placement, command, seen):
    """Cycles a ``cache[pointer]`` register holds on THIS pass, or None.

    The counting-round idiom (resonator_number_measurement): a DSP pointer is loaded with the
    cache base plus a word index, configured ``P+1``, advanced once per pass by ``pulse_cep()``,
    and the loop exits when it reaches a register holding base + index + rounds. The body reads
    its per-round value with ``bus_read(pointer)``. So pass r reads word ``index + r`` -- every
    term of which the program carries:

      * the pointer's starting address is a compile-time immediate (register_immediates),
      * the cache base is a firmware constant (cache_base),
      * and r is how many times this command has already been laid down.

    WHICH pointer feeds the register is not in the compiled record -- ``bus_read(pointer)`` emits
    ``BUS_ADDR <- DSP_P`` and the minor field there is the bus port, not the counter. It is taken
    from the LOOP the command sits in instead: the exit condition names the counter, and in this
    idiom the counter IS the pointer. When the enclosing loop's counter is not a cache pointer the
    link is unknown and this returns None, so the length stays honestly indeterminate rather than
    being resolved from a guess.
    """
    cache, base = trace.point_cache, trace.cache_base
    if not cache or base is None:
        return None
    if (trace.registers or {}).get(command.symbolic, {}).get("source") != "cache[pointer]":
        return None
    for context in reversed(getattr(placement, "conditional", ()) or ()):
        if context.get("kind") != "repeat_until":
            continue
        match = _LOOP_COUNTER_RE.search(context.get("condition") or "")
        if not match:
            continue
        start = (trace.register_immediates or {}).get(match.group(1))
        if start is None:
            continue
        word = start - base
        if word < 0:
            continue                     # the counter is not a cache pointer at all
        key = (command.channel, command.symbolic)
        value = cache.get(word + seen.get(key, 0))
        if value is None:
            return None
        seen[key] = seen.get(key, 0) + 1
        return int(value)
    return None


def _register_gate(trace, command, gate_index):
    """Decode a gate word latched through a register, or None if this is not one.

    The multi-rail XEB runtimes issue each gate as ``schedule_direct(channel, regs[n])`` after
    ``regs[n].load(bus_read(pointers[n]))``, so the compiled command is ``REGn -> BUS_DATA``
    rather than the ``BUS_DATA -> BUS_DATA`` form describe_cache_stream recognises. The word
    itself is in the captured cache, so the gate is fully recoverable.

    WHICH cache region belongs to WHICH channel is not stated anywhere in the program -- the
    pointer DSP's index is not in the decoded record. It is therefore established from the DATA
    and checked, not guessed: a region belongs to a channel only if its word decodes to an
    address that names a pulse ON THAT CHANNEL (``addr_names``). A rail's gates live at
    addresses that resolve for its own DAC and nowhere else, so a wrong pairing simply fails to
    resolve and is rejected.
    """
    symbolic = command.symbolic
    if not (symbolic and str(symbolic).startswith("REG")):
        return None
    # A CONTINUATION command's symbolic value is a LENGTH, not a command word. `use_stretch`
    # splits a pulse into ARB / CONST_CONT / ARB_CONT and puts the register into the MIDDLE
    # command's length field, so `symbolic` there names a hold length -- and decoding a length as
    # a packed `(address << 16) | (length - 1)` invents both an address and a wrong duration.
    # resonator_number_measurement's counting rounds are that shape: every ladder length was drawn
    # one cycle too long, read one cache word too early, and attributed to a pulse the cache word
    # never named. It looked plausible, which is what made it worth guarding rather than noticing.
    if command.kind in ("CONST_CONT", "ARB_CONT"):
        return None
    source = (trace.registers or {}).get(symbolic, {}).get("source")
    if source != "cache[pointer]":
        return None
    cache, names = trace.point_cache or {}, trace.addr_names or {}
    if not cache or not names:
        return None
    try:
        channel_num = int("".join(ch for ch in command.channel if ch.isdigit()))
    except ValueError:
        return None

    starts = trace.register_stream_starts
    if command.channel not in starts:
        # first gate on this channel: find the cache offset whose word names a pulse here
        for offset in sorted(cache):
            word = int(cache.get(offset, 0))
            if not word:
                continue
            if names.get((channel_num, word >> 16)) is not None:
                starts[command.channel] = offset
                break
        else:
            return None

    offset = starts[command.channel] + gate_index.get(command.channel, 0)
    word = int(cache.get(offset, 0))
    if not word:
        return None
    address = word >> 16
    io_name, pulse = names.get((channel_num, address), (None, None))
    if pulse is None:
        return None
    return {"length": (word & 0xFFFF) + 1, "address": address,
            "pulse": pulse, "io_name": io_name}


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

    WHICH drain it is matters, and the two are not interchangeable. The status bit polled is
    written to ``MASK`` by the instruction just before the hold:

    ==========  ====================  =====================================================
    ``MASK``    primitive             releases when
    ==========  ====================  =====================================================
    ``0x1``     ``dma_running``       (blocking poll -- not a drain at all)
    ``0x2``     ``fifo_empty``        the LAST descriptor is pulled
    ``0x8``     ``fifo_almost_empty`` ONE DESCRIPTOR EARLIER than that
    ==========  ====================  =====================================================

    The two compile to byte-identical poll instructions -- only this mask differs -- so a model
    that keys off the poll alone cannot tell them apart, and treating ``almost_empty`` as
    ``fifo_empty`` puts every following block one descriptor late.

    Both the masks and the one-descriptor offset are the firmware's, not a fitted constant.
    ``acadia_dma.vhd`` publishes the DMA status bus as ``miso(1) <= fifo_empty`` (mask ``0x2``)
    and ``miso(3) <= fifo_almost_empty`` (mask ``0x8``), which is what
    ``Acadia.channel_is_fifo_empty`` / ``channel_is_fifo_almost_empty`` read; the FIFO is an XPM
    macro with ``USE_ADV_FEATURES`` bit 11 enabled, whose ``almost_empty`` asserts while ONE word
    is still queued. One descriptor earlier, exactly. The loopback then measures that offset
    independently: ``batch_drain_almost`` ran 119.92 ns (24 cycles = one 120 ns descriptor) ahead
    of the old prediction per drain, and 239.97 ns over two.

    This is not a corner case. An audit of the 121 qudit-branch runtimes found
    ``channel_is_fifo_empty`` in *none* of them and ``channel_is_fifo_almost_empty`` in all
    seven that stream (dualrail_rb, xeb_1DR/2DR/3DR, beamsplitter_amp_detune_calibration):
    they refill the FIFO while it is still playing, which is the whole point of the primitive.

    :return: ``{nth: {"issue": cycles, "almost_empty": bool}}``
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
                    # the nearest preceding MASK write is the status bit this poll tests
                    mask = next((prog[k].imm1 for k in range(j - 1, trigger - 1, -1)
                                 if prog[k].d1 == "MASK"), None)
                    drains[nth] = {"issue": (nxt - j + 1) if nxt is not None else 0,
                                   "almost_empty": mask == ALMOST_EMPTY_MASK}
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
    self.length_underflows = []      # per-layout, like the two above
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
    # how many register-sourced gates have already been laid on each channel, so consecutive
    # plays read consecutive cache words the way the walking pointer does
    gate_index = {}
    # how many times each (channel, register) pointer read has been laid down, which IS the pass
    # index the walking pointer is on -- same bookkeeping as gate_index, for lengths
    pointer_reads = {}
    cursor, t_seq = {}, 0
    # Instruction span of the gap the sequencer is currently paying, i.e. how long BEFORE t_seq
    # the next block's descriptors start being pushed. Needed to tell a seamless continuation
    # from a real bubble (see play_start below).
    pending_issue = 0
    # Per-channel FIFO state, which is what decides whether playback continues seamlessly.
    # A channel drained to `fifo_empty` has NOTHING queued -- it stops and restarts at the next
    # trigger. A channel drained to `fifo_almost_empty` still holds one descriptor, so it plays
    # on, and commands pushed before it finishes continue without a bubble. The state belongs to
    # the CHANNEL and outlives the drain block: in `batch_almost -> batch` an unrelated marker
    # block runs in between, and the batched channel is still playing throughout it.
    queued = {}
    # Which blocks execute at all, for edge_gap to discount a skipped `test` body. A property of
    # the PLAN, so it is computed once: rebuilding it per placement made the layout quadratic --
    # 3000 placements meant 9.1 million dict lookups and 3 seconds, and pinning a loop to 1000
    # passes took 12 s (100 000, which the panel's spin box allowed, never finished at all).
    executed_nths = {nth_of_block.get(entry[0]) for entry in plan}
    executed_nths.discard(None)
    # Blocks with NO trigger of their own -- a `channel_synchronizer(trigger=False)` only queues,
    # so nothing anchors it to a program address. `executed_nths` cannot speak for them, and a
    # `test` arm built that way is invisible to the branch accounting unless it is named
    # separately. Both are properties of the plan, so both are computed once (see the note above:
    # rebuilding per placement made the layout quadratic).
    executed_indices = {entry[0] for entry in plan}
    unanchored_blocks = [i for i, b in enumerate(self.blocks)
                         if nth_of_block.get(i) is None and (getattr(b, "conditional", ()) or ())]
    for step, (index, iteration, path) in enumerate(plan):
        block = self.blocks[index]
        placement = Placement(index=index, iteration=iteration, path=path,
                              trigger=block.trigger, blocking=block.blocking,
                              conditional=block.conditional)
        channels = {c.channel for c in block.commands}

        # Per-channel start: each channel plays its own FIFO from the trigger, so it
        # begins at whichever is later -- the sequencer reaching the block (t_seq) or
        # that channel finishing what it was already playing (cursor[c]). Lock-step
        # channels share one value -- a single block-wide start.
        # A channel that is STILL PLAYING when this block's descriptors are pushed continues
        # seamlessly -- the player pulls the next descriptor the moment it finishes the current
        # one, so playback resumes at `cursor`, not at the trigger. It only restarts at the
        # trigger if its FIFO had already run dry by the time the pushes arrived.
        #
        # The distinction is invisible after a `fifo_empty` drain (the FIFO is empty by
        # definition) but decides the answer after `fifo_almost_empty`, which releases the
        # sequencer while one descriptor is still queued. Measured on the loopback:
        # `batch_almost -> batch` predicted 725 ns and measured 720.0 -- the model inserted a
        # one-cycle bubble at cursor=157 / t_seq=158 that the hardware does not have. The same
        # pair with a plain `fifo_empty` drain has cursor=157 / t_seq=182, a genuine 25-cycle
        # bubble, and there the trigger IS the start -- which is why `batch -> batch` was right
        # all along and only the almost_empty variants drifted.
        push_start = t_seq - pending_issue
        play_start = {}
        for ch in channels:
            at = cursor.get(ch, 0)
            # STRICTLY inside the push window. A channel whose playout ends exactly ON
            # push_start has not overlapped the pushes at all -- it finished as they began, so
            # there is nothing queued behind it and it restarts at the trigger like any other.
            # That boundary is precisely what a loop back-edge produces: on re-entry the cursor
            # lands on push_start, and treating it as seamless shortened every pass after the
            # first by 6 cycles. Measured on batch_in_loop_almost (dualrail_rb's exact shape):
            # hardware is a uniform 390 ns/pass at loop_count 3,4,5, while `<=` gave 390 then
            # 360, 360, 360 -- a drift that grows with the loop count (+30, +60, +90 ns).
            seamless = queued.get(ch, False) and push_start < at < t_seq
            play_start[ch] = at if seamless else max(t_seq, at)
        stream_here = False
        # last descriptor length per channel, for the last-pulled drain time, and the one
        # before it -- `fifo_almost_empty` releases a descriptor earlier than `fifo_empty`,
        # so it needs both (see drain_block_issue)
        last_len = {}
        prev_len = {}
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
            # Commands already placed by a stream expansion: the repeat copies of the same word,
            # and the per-pass extras that were laid between the gates. Checked BEFORE the stream
            # branch, because a repeat copy IS a stream command and would otherwise expand the
            # whole train again -- five copies of a three-gate train gave 75 pulses, not 15.
            placed_by_stream = set()
            for command in group:
                if id(command) in placed_by_stream:
                    continue
                if self._is_stream_command(command):
                    stream_here = True
                    start_rel = per_channel.get(command.channel, t_sub)
                    base = play_start[command.channel]
                    # Everything else this subschedule puts on the stream's channel belongs to
                    # EVERY pass of the loop, not once after it: the loop body is its own
                    # subschedule holding the gate and whatever follows it (an inter-gate dwell).
                    # Laying those after the train, as before, drew one dwell at the end and left
                    # the gate period short by its duration.
                    extras = [c for c in group
                              if c.channel == command.channel
                              and c is not command
                              and not self._is_stream_command(c)]
                    extra_len = sum(int(c.length or 0) for c in extras)
                    # The same word issued more than once in a pass: dualrail_rb repeats a short
                    # half-swap so the loop can keep up. Every such command is a stream command
                    # on this channel, so counting them gives the repeat factor -- and they must
                    # be expanded ONCE with that factor, not once each, or the model puts the
                    # refill floor between copies that actually abut.
                    same = [c for c in group
                            if c.channel == command.channel and self._is_stream_command(c)]
                    end = self._expand_stream(command, placement, base + start_rel,
                                              per_pass_extra=extra_len, extras=extras,
                                              repeats=len(same))
                    placed_by_stream.update(id(c) for c in extras)
                    placed_by_stream.update(id(c) for c in same[1:])
                    end_rel = end - base
                    per_channel[command.channel] = end_rel
                    prev_len[command.channel] = 0
                    last_len[command.channel] = 0
                    sub_len = max(sub_len, end_rel - t_sub)
                    continue
                length = command.length
                resolution = command.resolution
                gate = _register_gate(self, command, gate_index)
                if gate is not None:
                    # A gate word latched through a REGISTER before being issued
                    # (`regs[n].load(bus_read(pointers[n]))` then `schedule_direct(ch, regs[n])`
                    # -- the multi-rail XEB idiom). The word is in the captured cache, so the
                    # gate's length AND identity are known; without this it stayed symbolic and
                    # drew as an indeterminate grey box captioned with the register name.
                    length, resolution = gate["length"], "cache"
                    command = replace(command, pulse=gate["pulse"], io_name=gate["io_name"],
                                      address=gate["address"])
                    gate_index[command.channel] = gate_index.get(command.channel, 0) + 1
                elif command.symbolic == "BUS_DATA" and (self.direct_words or {}).get(
                        command.channel) is not None and (self.point_cache or {}):
                    # A direct DMA command replayed from a FIXED cache address: the word is in
                    # the captured cache, so its length is known exactly rather than falling
                    # back to `resolve_indeterminate` (0). Acadia packs an arbitrary command as
                    # `(address << 16) | (length - 1)` (see compiled_log.parse), so the low 16
                    # bits plus one are the cycles. Leaving it at 0 collapsed the whole train --
                    # BeamsplitterAmpDetuneCalibration plays one 26-cycle word 64 times, and
                    # drawing 64 zero-length commands put the readout after it ~1.2 us early.
                    word = self.direct_words[command.channel]
                    dma = self.point_cache.get(word)
                    if dma is not None:
                        length = (int(dma) & 0xFFFF) + 1
                        resolution = "cache"
                    else:
                        length = command.length or 0
                elif command.symbolic:
                    override = self.register_overrides.get(command.symbolic)
                    resolved = self.register_cycles.get(command.symbolic)
                    if resolved is None and override is None:
                        # a per-pass pointer read: pass r takes cache word index + r
                        resolved = _pointer_length(self, placement, command, pointer_reads)
                    value = (override if override is not None else
                             resolved if resolved is not None else fallback)
                    length = int(value)
                    resolution = ("override" if override is not None
                                  else "cache" if resolved is not None
                                  else "fallback")
                    # ONLY when the value is genuinely KNOWN to be zero. An unresolved length
                    # also reads as 0 -- `fallback` is `resolve_indeterminate`, which defaults to
                    # 0 -- and that means "we could not tell", not "the register held zero".
                    # Treating those as underflows would draw a 21-second command on every trace
                    # with an indeterminate length, which is far more misleading than the bug
                    # being guarded against.
                    if length == 0 and resolution in ("cache", "override"):
                        # ZERO UNDERFLOWS -- and the board does something enormous, so the picture
                        # must not quietly show nothing. `Acadia.command_dma` emits `length - 1`
                        # (system.py), so a register holding 0 becomes -1, i.e. an all-ones length
                        # field: 2**16-1 cycles for an ARB command, 2**32-1 for a 32-bit DWELL or
                        # CONST_CONT -- 328 us and ~21 SECONDS respectively, per shot, instead of
                        # nothing at all.
                        #
                        # This is not hypothetical. dual_rail_ramsey._delay_cycles floors its
                        # register dwell at 1 cycle for exactly this reason and says so; a delay
                        # sweep that starts at 0 is otherwise a 21-second first point. Drawing it
                        # as a 0-length command would hide the single most confusing failure a
                        # length sweep can produce, so it is modelled and flagged instead.
                        width_bits = 16 if command.kind in ("ARB", "ARB_CONT") else 32
                        length = (1 << width_bits) - 1
                        resolution = "underflow"
                        self.length_underflows.append(
                            {"channel": command.channel, "kind": command.kind,
                             "register": command.symbolic, "cycles": length})
                    self.unresolved += 1
                start_rel = per_channel.get(command.channel, t_sub)
                placement.commands.append(replace(
                    command, start=play_start[command.channel] + start_rel,
                    length=length, resolution=resolution))
                per_channel[command.channel] = start_rel + length
                prev_len[command.channel] = last_len.get(command.channel, 0)
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

        # gap to whatever executes NEXT (loop back-edges included).
        # Computed for a drain block as well as a blocking one: a drain's issue span has the
        # same address-order-vs-execution-order problem every other edge has (below).
        drain_edge = None
        if (placement.blocking or nth_of_block.get(index) in drains) and step + 1 < len(plan):
            from_nth = nth_of_block.get(index)
            # The next block ANCHORED IN THE PROGRAM, not simply the next one executed. A
            # `channel_synchronizer(trigger=False)` block emits no DMA trigger, so it has no
            # trigger address and no entry here -- and the old lookup then found None and gave
            # up, charging NOTHING for the whole stretch: the queued block's own command pushes,
            # the `channel_trigger` bus write that fires them, and the condition-plus-branch of
            # every `test` arm skipped along the way. Measured on the loopback (test_chain,
            # trigger=False, arm 0 taken) the sequencer really spends 75 ns on that stretch with
            # no skips at all, plus 25 ns for each arm skipped, while the model predicted a flat
            # 305 ns for 1, 2, 4 and 8 arms alike. Looking ahead to the next block that DOES have
            # a trigger puts those instructions back inside one edge, which is where the hardware
            # pays them: the sequencer fetches straight through the queued blocks without
            # stopping, because there is nothing to stop for until something is triggered.
            to_nth, ahead = None, step + 1
            while ahead < len(plan):
                to_nth = nth_of_block.get(plan[ahead][0])
                if to_nth is not None:
                    break
                ahead += 1
            if from_nth is not None and to_nth is not None:
                # Which trigger-less `test` arms lie on THIS stretch, and which of them ran.
                # In program order, because the compiled branches are in program order too, so
                # the k-th such branch the path walk meets is the k-th arm here.
                arms = [i in executed_indices for i in unanchored_blocks
                        if index < i < plan[ahead][0]]
                edge = edge_gap(self.control_flow, from_nth, to_nth, executed_nths, arms)
                drain_edge = edge
                if edge and not placement.blocking:
                    pass          # a drain block carries no gap_after; it advances t_seq below
                elif edge:
                    detect = self.gap_terms.get("detect", 3)
                    propagate = self.gap_terms.get("propagate", 2)
                    total = (detect + edge["issue"] + propagate
                             + DMA_STATUS_REGISTER + edge["branch_penalty"])
                    placement.gap_after = total
                    pending_issue = edge["issue"]
                    placement.gap_breakdown = {
                        "total": total, "detect": detect, "issue": edge["issue"],
                        "propagate": propagate,
                        "status_register": DMA_STATUS_REGISTER,
                        "branch_penalty": edge["branch_penalty"],
                        "edge": edge["kind"]}

        # advance the sequencer clock to the next block
        nth = nth_of_block.get(index)
        if nth in drains and not stream_here:
            # a block=False drain releases when the LAST descriptor is pulled -- the end
            # of playout minus that last (still-playing) descriptor. The sequencer then
            # still spends one boundary gap issuing the next block's pushes+trigger, so
            # the next block plays at last-pulled + that gap, not immediately.
            #
            # `fifo_almost_empty` asserts ONE DESCRIPTOR EARLIER than `fifo_empty`, so it also
            # backs off the descriptor before the last one. Both compile to the same poll and
            # differ only in the polled MASK bit; see drain_block_issue for the measurement.
            drain = drains[nth]
            detect = self.gap_terms.get("detect", 3)
            propagate = self.gap_terms.get("propagate", 2)
            # Prefer the EXECUTED edge. drain_block_issue counts from the drain poll to the next
            # trigger in ADDRESS order, which is only the next executed block when control falls
            # straight through. Inside a loop the next block executed is the loop head, reached
            # backwards over the branch, so the address-order span charges the wrong instructions
            # -- one cycle too many per pass here, which a loop then multiplies (batch_in_loop
            # measured exactly 5.00 ns x (loop_count - 1): 5.05 / 10.12 / 15.12 / 19.97 / 25.01 ns
            # at loop_count 2..6). edge_gap already walks the taken path and charges the branch,
            # so it is the authority whenever it resolves the edge.
            issue = drain_edge["issue"] if drain_edge else drain["issue"]
            penalty = drain_edge["branch_penalty"] if drain_edge else 0
            pending_issue = issue
            # this block's channels are the ones the drain polled
            for ch in channels:
                queued[ch] = drain["almost_empty"]
            back = ((lambda ch: last_len.get(ch, 0) + prev_len.get(ch, 0))
                    if drain["almost_empty"] else (lambda ch: last_len.get(ch, 0)))
            pulled = [cursor[ch] - back(ch) for ch in channels]
            release = max([t_seq] + pulled)
            # `detect` is the latency between the FIFO status CHANGING and the sequencer seeing
            # it. When the drain condition is already satisfied at the moment the poll executes
            # there is no transition to wait for, and the poll costs one cycle less. That is only
            # reachable when the release time does not run past the sequencer -- i.e. the batch
            # was short enough that it had already emptied to the polled level.
            #
            # Measured directly by sweeping the descriptor count of an almost_empty drain
            # (batch_drain_almost, batch_resync_pulses 2..10): n=2 missed by exactly one cycle per
            # drain (-4.94 ns, then -9.97 ns cumulative over two), while n=3,4,5,6,8,10 all agreed
            # to <=0.19 ns. With 2 descriptors the "one word left" level is reached as soon as the
            # first is pulled, so the condition is true on arrival; with 3 or more the sequencer
            # genuinely waits.
            already_true = max(pulled, default=0) <= t_seq
            # The state register is charged only when there IS a transition to wait for; see
            # DMA_STATUS_REGISTER. Arithmetically identical to the two constants this replaces
            # (detect - 1 ... + 1), but it says which cycle of hardware is being counted.
            gap = (detect + issue + propagate + penalty
                   + (0 if already_true else DMA_STATUS_REGISTER))
            t_seq = release + gap
        elif placement.blocking or stream_here:
            # blocking block (or a cache stream whose post-loop drain holds the whole
            # sequencer): playout completes for every channel before the next block.
            t_seq = placement.stop + placement.gap_after
            for ch in channels:
                cursor[ch] = t_seq
                queued[ch] = False     # a blocking block waits for playout: nothing left queued
        else:
            # a plain non-blocking batch: t_seq unchanged, channels keep playing. No new gap was
            # paid, so there is no fresh push window either -- clear it rather than let the
            # previous block's issue span widen the seamless-continuation band below.
            pending_issue = 0

        self.placements.append(placement)
    return self
