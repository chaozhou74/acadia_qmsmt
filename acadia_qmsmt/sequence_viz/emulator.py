"""Instruction-level emulator of the Acadia sequencer.

SeeQuence's older layer recognises *idioms* in the compiled program (a cache-pointer stream, a
pointer pair, a cache sum) and resolves values only for the shapes it knows. A program written in
another shape is then misread -- the first MultiDualRailRB form (start AND end pointers loaded
from cache values) was read as a ~2-million-pass loop. This module does not recognise anything.
It EXECUTES the compiled STP program, instruction by instruction, for one sweep point's cache,
and reports what the sequencer does: every DMA command word written to every channel, every
trigger, every wait, in executed order. Whatever the runtime's code looks like, the answer comes
from the same few rules below.

The machine (acadia/sequencer.py, acadia/dsp_modes.py, Xilinx UG579):

* An STP instruction has two slots, ``dest1 <- src1`` and ``dest2 <- src2``. When it is
  ``conditional``, only slot 1 is written, and only if the condition holds. The condition reads
  ``src2``: optionally inverted (``premask_invert``), then ``AND`` / ``XOR`` with MASK (``op``),
  then compared with zero -- ``!= 0`` normally, ``== 0`` with ``condition_invert``.
* Sources: REG0..7, IMM (the immediate), BUS_DATA (the word read from the address last put on
  BUS_ADDR), DSP_P (a DSP slice's output), STACK.
* Destinations: REG, MASK, PC (absolute branch / hold, relative hold = halt), BUS_ADDR (alone:
  address a READ; together with BUS_DATA in the same instruction: a WRITE), DSP_AB / DSP_C (the
  slice's inputs), DSP_CFG (its mode word).
* A DSP slice computes ``P <- f(W, X, Y, Z, CIN)`` with the operands chosen by its opmode and the
  operation by its ALU mode (UG579 table 2-7/2-8). It computes when its clock enable fires:
  once when a configuration with ``dsp_cep="pulse"`` is written, once per ``pulse_cep()``
  (the instruction's ``dsp_cep`` field), or every cycle with ``dsp_cep="set"``.
* The bus map is the firmware's ``sequencer_bus_decoder``: DMA command ports per channel
  (``base + command type``), the DMA trigger, the DMA running/FIFO status, module registers
  (CMACC results), the cache, ...

What the emulator does NOT know, and says so rather than guessing:

* a value only the board has (a measurement result). A runtime `test` / `loop` / `repeat_until`
  on one is decided by :class:`ControlFlow`: a pin (``loop_counts`` / ``path_choices``) if there
  is one, else the unknown is read as 0 for a test and one pass for a loop; the construct is then
  listed as assumed, and every placement after it carries ``assumed=True``. Any other branch on
  such a value takes ``choices`` ({program address: taken?}) or the value 0.
* a DSP counting freely (``dsp_cep="set"``): its value depends on time, so it reads as unknown.

Time IS modelled: one cycle per instruction plus the measured terms in :func:`default_timing`, and
a per-channel DMA FIFO model (:class:`DmaChannel`). See the README section "Instruction-level
emulator" for each rule and the board case behind it. PC and stack behaviour follow the firmware
(acadia/logic/src/acadia_sequencer.vhd): a hold that held resumes at its source address, a
``push_return`` pushes whether or not its branch is taken, and a return from an empty stack is
reported, not guessed.
"""
from dataclasses import dataclass, field
from typing import Optional

import re

from acadia.sequencer import Destination, Source

from .machine import ONE_CYCLE_JOIN_EARLY
from .tracing import DMA_STATUS_REGISTER, MEASURED_BRANCH_PENALTY, MEASURED_CMACC_RESULT_LATENCY

MASK32 = 0xFFFFFFFF
MASK48 = (1 << 48) - 1

# Destination / source major codes, by name (acadia.sequencer.Destination.Major / Source.Major)
D = Destination.Major
S = Source.Major


@dataclass
class Op:
    """One instruction as the hardware holds it: immediates are unsigned 32-bit words."""
    address: int
    d1: str
    d1_minor: int
    s1: str
    s1_minor: int
    imm1: int
    d2: str
    d2_minor: int
    s2: str
    s2_minor: int
    imm2: int
    conditional: bool
    condition_invert: bool
    premask_invert: bool
    op: Optional[str]
    cep: Optional[int]          # DSP whose clock enable this instruction pulses
    push_return: bool
    comment: Optional[str]


def compile_program(acadia):
    """The sequencer program as a list of :class:`Op`, index = program address.

    It is decoded from the ASSEMBLED bytes (:func:`assembled_bytes`, :func:`decode_binary`), so
    the emulator runs exactly the instruction memory the board is given: every immediate as the
    32-bit word it becomes, and every instruction the assembler turns into a NOP (a DMA command
    whose word is -1, acadia's dwell optimisation, ``Acadia.sequencer_dma_instruction_simplifier``)
    as that NOP. Checked 2026-11-18: for all 11 selftest folders the re-traced bytes equal the
    archived ``assembled.log`` byte for byte (validation/emulator_vs_trace.py --binary).
    The compiler's comments ride along for display.
    """
    sequencers = acadia._sequencer_type.instances
    if len(sequencers) != 1:
        raise NotImplementedError(f"{len(sequencers)} sequencer instances; the emulator runs one")
    comments = {a: ins.comment for a, ins in enumerate(sequencers[0]._compiled_program)}
    return decode_binary(assembled_bytes(acadia), comments)


def decode_binary(blob, comments=None):
    """The program as a list of :class:`Op`, decoded from the ASSEMBLED bytes -- what the board's
    instruction memory holds (``Acadia.assemble``; an archived run's ``assembled.log``).

    This is the inverse of ``STP.assemble``: 16 bytes per instruction, ``<IIQ`` = imm2, imm1,
    then the control word
        bit 48 conditional, 40 push_return, 32-39 src1, 24-31 src2, 16-23 dest1, 8-15 dest2,
        7 condition_invert, 6 op=and, 5 op=xor, 4 premask_invert, 3 dsp_cep enable, 0-2 its DSP.
    A port byte is ``major << 3 | minor``. An all-zero word is a NOP (``REG0 -> NONE``), which is
    what the simplifier leaves in place of a removed instruction.
    ``comments``: optional {address: text} to carry the compiler's comments along.
    """
    import struct
    if isinstance(blob, str):
        blob = bytes.fromhex(blob)
    program = []
    for address in range(len(blob) // 16):
        imm2, imm1, ctl = struct.unpack_from("<IIQ", blob, 16 * address)

        def port(byte, kind):
            return kind((byte >> 3) & 0x1F).name, byte & 0x7

        s1, s1_minor = port((ctl >> 32) & 0xFF, S)
        s2, s2_minor = port((ctl >> 24) & 0xFF, S)
        d1, d1_minor = port((ctl >> 16) & 0xFF, D)
        d2, d2_minor = port((ctl >> 8) & 0xFF, D)
        program.append(Op(
            address=address, d1=d1, d1_minor=d1_minor, s1=s1, s1_minor=s1_minor, imm1=imm1,
            d2=d2, d2_minor=d2_minor, s2=s2, s2_minor=s2_minor, imm2=imm2,
            conditional=bool((ctl >> 48) & 1), condition_invert=bool((ctl >> 7) & 1),
            premask_invert=bool((ctl >> 4) & 1),
            op="and" if (ctl >> 6) & 1 else ("xor" if (ctl >> 5) & 1 else None),
            cep=(ctl & 0x7) if (ctl >> 3) & 1 else None,
            push_return=bool((ctl >> 40) & 1),
            comment=(comments or {}).get(address)))
    return program


def assembled_bytes(acadia):
    """The sequencer's instruction memory exactly as ``Acadia.assemble`` would write it."""
    sequencers = acadia._sequencer_type.instances
    if len(sequencers) != 1:
        raise NotImplementedError(f"{len(sequencers)} sequencer instances; the emulator runs one")
    return b"".join(ins.assemble() for ins in sequencers[0]._compiled_program)


def load_assembled_log(path):
    """The program of an archived run, from its ``assembled.log`` (folder or file)."""
    import json
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        path = path / "assembled.log"
    segments = json.loads(path.read_text())
    if list(segments) != ["seq@00000000"]:
        raise NotImplementedError(f"segments {list(segments)}; the emulator runs one sequencer")
    return decode_binary(segments["seq@00000000"])


def bus_map(acadia):
    """``[(start, size, device name)]`` from the firmware's sequencer bus decoder."""
    decoder = acadia._firmware.sequencer_bus_decoder
    return sorted((int(device.address().value()), int(device.size), device.name)
                  for device in decoder)


def _device(bus, address):
    """``(device name, offset)`` for a bus address, or ``(None, address)``.

    A device's window runs from its base to the next device's base. ``size`` is NOT the window:
    a DMA port has size 2 yet takes its four command types at base + 0..3 (Acadia.command_dma),
    and bounding by size dropped every continued-constant and dwell command."""
    for k, (start, size, name) in enumerate(bus):
        end = bus[k + 1][0] if k + 1 < len(bus) else start + max(size, 1)
        if start <= address < end:
            return name, address - start
    return None, address


# ---------------------------------------------------------------------------------------------
# DSP slice (UG579). The configuration word is acadia's DSPConfiguration packing:
#   bits 15-16 cep (0 reset, 1 set, 3 pulse), 14 rst_p, 13 cin, 4-12 opmode, 0-3 alumode,
#   opmode = w << 7 | z << 4 | y << 2 | x.
# ---------------------------------------------------------------------------------------------

def decode_dsp_config(word):
    """The fields of a DSP configuration word."""
    opmode = (word >> 4) & 0x1FF
    return {"cep": (word >> 15) & 0x3, "rst_p": bool((word >> 14) & 1), "cin": (word >> 13) & 1,
            "w": (opmode >> 7) & 0x3, "z": (opmode >> 4) & 0x7, "y": (opmode >> 2) & 0x3,
            "x": opmode & 0x3, "alumode": word & 0xF}


@dataclass
class DSPSlice:
    ab: Optional[int] = 0
    c: Optional[int] = 0
    p: Optional[int] = 0
    config: dict = field(default_factory=lambda: decode_dsp_config(0))
    counting: bool = False      # dsp_cep="set": P changes every cycle, so its value is time


def dsp_output(cfg, ab, c, p, pcin):
    """``P`` after one enabled clock, per UG579's multiplexers and ALU modes; None if an operand
    it uses is unknown."""
    def need(v):
        if v is None:
            raise _Unknown
        return v
    try:
        # X: 0 / P / A:B      Y: 0 / all ones / C      W: 0 / P / C
        # Z: 0 / PCIN (the slice below) / P / C / P >> 17
        x = need(p) if cfg["x"] == 2 else (need(ab) if cfg["x"] == 3 else 0)
        y = need(c) if cfg["y"] == 3 else (MASK48 if cfg["y"] == 2 else 0)
        z = 0
        if cfg["z"] == 1:
            z = need(pcin)
        elif cfg["z"] == 2:
            z = need(p)
        elif cfg["z"] == 3:
            z = need(c)
        elif cfg["z"] == 6:
            z = need(p) >> 17
        w = need(p) if cfg["w"] == 1 else (need(c) if cfg["w"] == 3 else 0)
    except _Unknown:
        return None
    alu, cin = cfg["alumode"], cfg["cin"]
    if alu == 0b0000:
        out = z + w + x + y + cin
    elif alu == 0b0011:
        out = z - (w + x + y + cin)
    elif alu == 0b0001:
        out = -z + (w + x + y + cin) - 1
    elif alu == 0b0010:
        out = -(z + w + x + y + cin) - 1
    elif alu in (0b0100, 0b0101, 0b0110, 0b0111, 0b1100, 0b1101, 0b1110, 0b1111):
        # two-input logic on X and Z. ALU mode bit 0 inverts Z (acadia's `alumode + inv_z`);
        # Y = 0 selects XOR/XNOR/AND/NAND, Y = all ones OR/NOR.
        if alu & 1:
            z = z ^ MASK48
        base = alu & 0b1110
        if base == 0b0100:
            out = x ^ z
        elif base == 0b0110:
            out = ~(x ^ z)
        elif base == 0b1100:
            out = (x | z) if cfg["y"] == 2 else (x & z)
        else:
            out = ~(x | z) if cfg["y"] == 2 else ~(x & z)
    else:
        return None
    return out & MASK48


def _signed32(value):
    value &= MASK32
    return value - (1 << 32) if value >> 31 else value


class _Unknown(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Timing constants. Every one is a firmware fact or a loopback measurement already used by the
# older layout (tracing.py / machine.py); the clock below only applies them per instruction.
# ---------------------------------------------------------------------------------------------

#: A taken branch flushes the two-stage fetch pipeline and spends one cycle writing the PC
#: (acadia_sequencer.vhd instruction_proc); confirmed on the loopback (tracing.py).
BRANCH_PENALTY = MEASURED_BRANCH_PENALTY
#: A descriptor written to a DMA command port can be pulled this many cycles after the write
#: instruction. Fixed by the measured seamless/bubble boundary (machine.py, play_start).
PUSH_TO_FIFO = 2


def default_timing(detect=3, propagate=2):
    """The timing terms, from the firmware's bus pipeline (SequenceTrace.gap_terms).

    status_lag   a hold or read sees a DMA status bit as it was this many cycles earlier: the read
                 path (``Acadia._bus_latency("dma_running")``, ``detect``) plus the DMA's own state
                 register (DMA_STATUS_REGISTER, tracing.py).
    trigger      the first sample of a triggered channel leaves this many cycles after the trigger
                 instruction: the trigger dataport pipeline (``propagate``) plus the write cycle.
    cmacc        a CMACC "measurement complete" poll releases this many INSTRUCTION cycles after
                 the end of its capture window. MEASURED_CMACC_RESULT_LATENCY (11) was measured
                 against the older layout's poll time, which it places ``propagate`` cycles after
                 the instruction cycle (docs/EMPIRICAL_CONSTANTS.md section 3), so here it is
                 11 - propagate = 9. Checked on the board recording it was measured with
                 (gkp/cmacc_feedback_board_check.py --saved 261101_005015): 20 marker intervals.
    """
    trigger = propagate + 1
    return {"status_lag": detect + DMA_STATUS_REGISTER, "trigger": trigger,
            "cmacc": MEASURED_CMACC_RESULT_LATENCY - propagate, "push": PUSH_TO_FIFO,
            "branch": BRANCH_PENALTY, "one_cycle_join_early": ONE_CYCLE_JOIN_EARLY}


# ---------------------------------------------------------------------------------------------
# DMA channels
# ---------------------------------------------------------------------------------------------

@dataclass
class Played:
    """One DMA descriptor: pushed by the instruction at ``push_pc``, played from ``start``."""
    channel: str
    ctype: int
    word: Optional[int]
    push_pc: int
    push_time: int
    avail: int                         # first cycle it can be pulled
    start: Optional[int] = None        # None while it waits for a trigger
    pulled: Optional[int] = None       # when it left the FIFO (see DmaChannel)
    trigger_step: Optional[int] = None  # the trigger that started its run (block identity)

    @property
    def length(self):
        if self.word is None:
            return 0
        return (self.word & 0xFFFF) + 1 if self.ctype == 1 else self.word + 1

    @property
    def stop(self):
        return None if self.start is None else self.start + self.length


class DmaChannel:
    """One channel's DMA: a FIFO of descriptors and a player (acadia_dma.vhd).

    * A pushed descriptor waits in the FIFO. A TRIGGER starts an idle channel: it plays its queued
      descriptors back to back, from ``trigger`` cycles after the trigger instruction.
    * While it plays, a descriptor that is in the FIFO when the current one ends is pulled at once
      -- playback continues with no gap and needs no new trigger. If the FIFO is empty then, the
      channel stops; anything pushed later waits for the next trigger.
    * A descriptor leaves the FIFO when the player PULLS it, so ``fifo_empty`` asserts while the
      last descriptor is still playing, ``fifo_almost_empty`` one descriptor earlier. The FIRST
      descriptor of a run is pulled by the trigger itself (it then needs ``trigger`` cycles to
      reach the output); each later one when the one before it ends. This is what the loopback
      shows: a drain whose batch is one descriptor (or two, for almost_empty) is already
      satisfied when the poll arrives, while with more descriptors the poll releases
      ``status_lag`` cycles after the last one starts (machine.py, batch_drain_almost and the
      join_marker cases).
    * ``running`` is set by the trigger and cleared when the last descriptor ends.
    """

    def __init__(self, name, timing):
        self.name = name
        self.timing = timing
        self.descriptors = []          # every descriptor, in push order
        self.pending = []              # pushed, waiting for a trigger
        self.chain_end = None          # end of the last descriptor scheduled
        self.runs = []                 # [trigger time, end] of each run (``running``)
        self.run_members = []          # the descriptors each run played
        self.last_trigger = None       # cycle of the latest trigger naming this channel
        # Speed only, no physics: the status queries below scan only what can still change a
        # flag. Without this every hold rescans every descriptor ever pushed, and a long run (a
        # streamed loop of thousands of passes) costs time ~ descriptors^2 per hold.
        #   started  descriptors given a start, in start order (starts never go backwards)
        #   pending  pushed, waiting for a trigger (above)
        #   orphans  dropped from pending by a DMA reset but never started; the status flags
        #            have always counted them as queued, so they stay in the scan
        # Everything in ``started`` before ``settled_started`` (and in ``runs`` before
        # ``settled_runs``) is finished by cycle ``settled_at``: all its times are <= it.
        self.started = []
        self.orphans = []
        self.settled_at = -1
        self.settled_started = 0
        self.settled_runs = 0

    def settle(self, t):
        """Advance the finished-prefix cursors to cycle ``t`` (never backwards)."""
        if t <= self.settled_at:
            return
        ds, k = self.started, self.settled_started
        while k < len(ds) and max(ds[k].avail, ds[k].start + 1, ds[k].pulled) <= t:
            k += 1
        runs, r = self.runs, self.settled_runs
        while r < len(runs) and runs[r][1] <= t:
            r += 1
        self.settled_started, self.settled_runs, self.settled_at = k, r, t

    def _live(self, tau):
        """(descriptors, runs) that can still change a flag at cycle ``tau`` or later."""
        if tau >= self.settled_at:
            return (self.started[self.settled_started:] + self.pending + self.orphans,
                    self.runs[self.settled_runs:])
        return self.descriptors, self.runs

    def push(self, played):
        self.descriptors.append(played)
        if self.chain_end is not None and played.avail < self.chain_end and not self.pending:
            # the player is still busy when it arrives: it follows on seamlessly. STRICTLY
            # before the current descriptor ends -- one arriving on the very cycle the player
            # finishes is too late, the channel stops and the descriptor waits for the next
            # trigger (loopback: batch_uneven's last pulse, 6 cycles later; machine.py's
            # batch_in_loop_almost boundary)
            played.start = played.pulled = self.chain_end
            self.started.append(played)
            played.trigger_step = self.descriptors[-2].trigger_step if len(self.descriptors) > 1 else None
            self.chain_end = played.stop
            self.runs[-1][1] = self.chain_end
            self.run_members[-1].append(played)
        else:
            self.pending.append(played)

    def trigger(self, t, step):
        """The trigger instruction executes at cycle ``t``."""
        self.last_trigger = t
        first = t + self.timing["trigger"]
        if self.chain_end is not None and self.chain_end > first:
            return                     # already running: the trigger changes nothing
        # Only a descriptor already IN the FIFO is started. One pushed a cycle before the trigger
        # has not arrived yet (``push``), so a trigger right behind its push starts nothing and
        # the channel waits for the next trigger -- which is why acadia puts "Delay accounting
        # for DMA FIFO latency" NOPs before every block's trigger. Measured on the loopback:
        # test_chain trigger_false_n1 (push, then channel_trigger at the very next instruction)
        # played 5 cycles later than an immediate start, exactly at the following trigger.
        ready = [p for p in self.pending if p.avail <= t]
        if not ready:
            return
        later = self.pending[len(ready):]
        start = first
        self.runs.append([t, start])
        self.run_members.append(list(ready))
        for k, p in enumerate(ready):
            p.start, p.trigger_step = start, step
            p.pulled = t if k == 0 else start
            self.started.append(p)
            start = p.stop
        self.pending = []
        self.chain_end = start
        self.runs[-1][1] = start
        for p in later:                # arrive while the run plays: they follow on
            self.descriptors.remove(p)
            self.push(p)

    def reset(self):
        self.orphans += self.pending
        self.pending = []

    # status, as the hardware's flags read at cycle tau
    def running(self, tau):
        """Set by the trigger, cleared when the last descriptor of the run ends. (The DMA's own
        state register delays the flag one more cycle; that cycle is in ``status_lag``.)"""
        return any(a <= tau < b for a, b in self._live(tau)[1])

    def queued(self, tau):
        return sum(1 for d in self._live(tau)[0]
                   if d.avail <= tau and (d.pulled is None or d.pulled > tau))

    def status_word(self, tau):
        """``Acadia.dma_status``: bit 0 running, 1 FIFO empty, 3 FIFO almost empty."""
        q = self.queued(tau)
        return (int(self.running(tau)) | (int(q == 0) << 1) | (int(q <= 1) << 3))

    def change_times(self, after):
        """Cycles after ``after`` at which any flag can change."""
        out = set()
        self.settle(after)             # all times before ``after`` are dropped below anyway
        descriptors, runs = self._live(after)
        for a, b in runs:
            out.update((a, b))
        for d in descriptors:
            out.add(d.avail)
            if d.start is not None:
                out.update((d.start, d.start + 1, d.pulled))
        return {x for x in out if x > after}


# ---------------------------------------------------------------------------------------------
# Control-flow constructs: which branch decides each `test` / `loop` / `repeat_until`
# ---------------------------------------------------------------------------------------------

@dataclass
class Construct:
    """One ``test`` / ``loop`` / ``repeat_until`` block of the runtime, located in the program.

    acadia compiles them as (acadia/sequencer.py):
      loop, repeat_until   body, then ONE conditional branch back to the body's first
                           instruction (``head``) -- a do-while: taken = run another pass
      test                 ONE conditional branch first. speculation=True: the body is inline and
                           the branch jumps PAST it, so the body runs when it is NOT taken.
                           speculation=False: the body is out of line and the branch jumps TO it
                           (pushing the return address), so the body runs when it IS taken.
    """
    id: int
    kind: str                   # "test" | "loop" | "repeat_until"
    level: int                  # one-based nesting level (SequenceTrace.construct_key)
    first: int                  # first block of the body (how the trace names the construct)
    branch: int                 # program address of the deciding branch
    head: Optional[int]         # loop: where a pass starts (the back branch's target)
    body: frozenset             # program addresses inside the construct (out-of-line too)
    speculation: Optional[bool] = None


def construct_map(extents, blocks, program):
    """``{branch address: Construct}`` from the recorder's extents (dryrun.branch_recorder).

    ``extents[id] = (acadia Instruction list, first, end)``: the instructions each construct
    created, which after compile carry their program addresses. Only constructs that contain a
    traced block are kept -- those are the ones the drawing names (``block.conditional``)."""
    named = {}
    for index, block in enumerate(blocks):
        for depth, context in enumerate(block.conditional or ()):
            if context.get("id") is not None and context["id"] not in named:
                named[context["id"]] = (index, depth + 1, context)
    out = {}
    for cid, (first, level, context) in named.items():
        if cid not in extents:
            continue
        instructions, k0, k1 = extents[cid]
        spans = []
        for k in range(k0, k1):
            ins = instructions[k]
            if not ins.address.assigned():
                continue
            a = ins.address.value()
            spans.append((k, a, a + len(ins.compiled)))
        if not spans:
            continue
        kind = context.get("kind")
        # the deciding instruction: a test's branch is the first thing it creates, a loop's the
        # last (created after its body); either may compile to a MASK load plus the branch
        _k, lo, hi = spans[0] if kind == "test" else spans[-1]
        branch = next((a for a in range(lo, hi) if a < len(program) and program[a].d1 == "PC"
                       and program[a].conditional and program[a].d1_minor in (
                           Destination.PC_ABSOLUTE_BRANCH, Destination.PC_RELATIVE_BRANCH)), None)
        if branch is None:
            continue
        head = None
        if kind == "test":
            body = {a for k, s, e in spans[1:] for a in range(s, e)}
        else:
            op = program[branch]
            if op.s1 != "IMM":
                continue
            head = (op.imm1 if op.d1_minor == Destination.PC_ABSOLUTE_BRANCH
                    else branch + _signed32(op.imm1))
            # instructions created before the head (a loop's DSP set-up) run once, outside it
            k_head = next((k for k, s, e in spans if s <= head < e), k0)
            body = {a for k, s, e in spans if k >= k_head for a in range(s, e)}
        out[branch] = Construct(cid, kind, level, first, branch, head, frozenset(body),
                                context.get("speculation"))
    return out


def trigger_blocks(raw_blocks, program, bus):
    """``{trigger program address: block index}``. Each traced block records the acadia
    instructions it compiled to (tracing.trace_runtime); its trigger is the one among them that
    writes the DMA trigger. Numbering triggers in ADDRESS order instead is wrong as soon as a
    block sits out of line (``test(speculation=False)``): its trigger is then at the end of the
    program while the block is numbered where the runtime wrote it."""
    trigger_address = next((start for start, _size, name in bus if name == "dma_trigger"), None)
    out = {}
    for index, raw in enumerate(raw_blocks):
        if not raw.get("trigger") or "instructions" not in raw:
            continue
        instructions, k0, k1 = raw["instructions"]
        for ins in instructions[k0:k1]:
            if not ins.address.assigned():
                continue
            a = ins.address.value()
            for address in range(a, a + len(ins.compiled)):
                op = program[address] if address < len(program) else None
                if (op is not None and op.d1 == "BUS_ADDR" and op.s1 == "IMM"
                        and op.imm1 == trigger_address and op.d2 == "BUS_DATA"):
                    out[address] = index
    return out


class ControlFlow:
    """Follows the constructs while :func:`execute` runs, and applies the user's pins.

    For every executed instruction that is a construct's deciding branch it decides the branch
    (from the value, a pin, or the default for a board-decided value) and records it, so the
    layout can say for every trigger WHICH pass of every enclosing construct it belongs to
    (``path``, as SequenceTrace numbers it) and how many passes / which arm every EXECUTION took.

    ``count_pin(first, level, path)`` / ``arm_pin(first, level, path)``: the trace's pins
    (SequenceTrace._count_override / _path_override); None = not pinned.
    """

    def __init__(self, constructs, count_pin=None, arm_pin=None):
        self.by_branch = constructs
        self.count_pin = count_pin or (lambda *a: None)
        self.arm_pin = arm_pin or (lambda *a: None)
        self.loops_at_head = {}
        for c in constructs.values():
            if c.kind != "test" and c.head is not None:
                self.loops_at_head.setdefault(c.head, []).append(c)
        for v in self.loops_at_head.values():
            v.sort(key=lambda c: c.level)
        self.watch = set(self.loops_at_head)
        self._stacks = {}
        self.passes = {c.id: 0 for c in constructs.values()}
        #: {(first, level): {enclosing path: passes (loop) or body ran? (test)}}
        self.executions = {}
        #: constructs decided at least once on a value the board decides, by (first, level)
        self.unknown = set()
        self.pinned = set()
        self.last_branch = None          # address of the branch taken to reach the current pc
        self.last_decided = None         # the construct decided most recently

    def stack(self, pc):
        """Constructs containing ``pc``, outermost first."""
        hit = self._stacks.get(pc)
        if hit is None:
            hit = self._stacks[pc] = sorted((c for c in self.by_branch.values() if pc in c.body),
                                            key=lambda c: c.level)
        return hit

    def path(self, pc):
        """Pass index of every construct containing ``pc`` (a test counts as pass 0)."""
        return tuple(0 if c.kind == "test" else self.passes[c.id] for c in self.stack(pc))

    def enclosing(self, c):
        return tuple(0 if o.kind == "test" else self.passes[o.id]
                     for o in self.stack(c.branch) if o.level < c.level)

    def enter(self, pc):
        """At a loop head: a fresh execution unless we came back over a branch inside the loop.
        Returns a new pc to skip the loop when its execution is pinned to 0 passes."""
        came_from = self.last_branch
        for c in self.loops_at_head[pc]:
            if came_from is not None and came_from in c.body:
                continue                                     # the next pass of a running loop
            self.passes[c.id] = 0
            outer = self.enclosing(c)
            pin = self.count_pin(c.first, c.level, outer)
            self.executions.setdefault((c.first, c.level), {})
            # 0 passes is offered for repeat_until only; a `loop` pin is at least one pass,
            # as SequenceTrace.execution_plan draws it
            if pin is not None and int(pin) <= 0 and c.kind == "repeat_until":
                self.pinned.add((c.first, c.level))
                self.executions[(c.first, c.level)][outer] = 0
                return c.branch + 1
        return None

    def decide(self, pc, value_holds, default_holds):
        """Decide construct ``pc``'s branch. ``value_holds``: from the data, None if the board
        decides it; ``default_holds``: what an unknown read as 0 gives."""
        c = self.by_branch[pc]
        key = (c.first, c.level)
        outer = self.enclosing(c)
        runs = self.executions.setdefault(key, {})
        self.last_decided = c
        if c.kind == "test":
            pin = self.arm_pin(c.first, c.level, outer)
            if pin is not None:
                self.pinned.add(key)
                body = bool(pin)
                holds = (not body) if c.speculation is not False else body
            else:
                holds = value_holds
                if holds is None:
                    self.unknown.add(key)
                    holds = default_holds
                body = (not holds) if c.speculation is not False else holds
            runs[outer] = body
            return holds
        done = self.passes[c.id] + 1                         # passes finished, this one included
        pin = self.count_pin(c.first, c.level, outer)
        if pin is not None:
            self.pinned.add(key)
            holds = done < (int(pin) if c.kind == "repeat_until" else max(int(pin), 1))
        elif value_holds is None:
            self.unknown.add(key)
            holds = False                # a board-decided count: one pass drawn, as labelled
        else:
            holds = value_holds
        if holds:
            self.passes[c.id] = done
        else:
            runs[outer] = done
        return holds


# ---------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------

@dataclass
class Event:
    """Something the sequencer did, in executed order, at cycle ``time``.

    kind  "dma"      a command word written to a channel's DMA (channel "DAC10" / "ADC6",
                     ctype 0 continued arbitrary, 1 arbitrary, 2 continued constant, 3 dwell)
          "trigger"  the DMA trigger written with ``value`` = channel mask
          "wait"     a PC hold on a peripheral, from ``time`` to ``end`` (``device``, ``mask``)
          "write"    any other bus write (``device``, ``offset``, ``value``)
    """
    kind: str
    address: int                # program address of the instruction
    step: int                   # executed-instruction index
    channel: Optional[str] = None
    ctype: Optional[int] = None
    value: Optional[int] = None
    device: Optional[str] = None
    offset: Optional[int] = None
    mask: Optional[int] = None
    time: int = 0
    end: Optional[int] = None
    #: trigger: pass index of every construct around it (ControlFlow.path), outermost first
    path: tuple = ()


@dataclass
class Execution:
    events: list = field(default_factory=list)
    steps: int = 0
    cycles: int = 0
    halted: bool = False
    #: {program address: [taken, ...]} for branches decided on an unknown value
    assumed: dict = field(default_factory=dict)
    #: executed-instruction index of the first such branch (None: the path is fully determined)
    first_assumed_step: Optional[int] = None
    #: bus reads of devices whose value is not known, {device: count}
    unknown_reads: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    #: {program address: times executed}
    visits: dict = field(default_factory=dict)
    #: where the step limit stopped a run that never halted (None if it halted)
    stop_pc: Optional[int] = None
    #: {channel: DmaChannel} -- every descriptor with the cycle it played
    channels: dict = field(default_factory=dict)
    #: the timing terms the clock used (default_timing)
    timing: dict = field(default_factory=dict)
    #: the ControlFlow that decided the runtime's constructs, if one was given
    control: Optional[object] = None

    def dma_words(self):
        """``{channel: [(ctype, word), ...]}`` in executed order."""
        out = {}
        for e in self.events:
            if e.kind == "dma":
                out.setdefault(e.channel, []).append((e.ctype, e.value))
        return out

    def played(self):
        """``{channel: [Played, ...]}`` in push order (``start`` None: never triggered)."""
        return {name: list(ch.descriptors) for name, ch in self.channels.items()}


def _channel_bit(channel):
    """Bit of ``channel`` in the dma_running / trigger masks: DACn -> n, ADCn -> 16 + n."""
    return int(channel[3:]) + (16 if channel.startswith("ADC") else 0)


def stream_sources(acadia):
    """``{"module<N>_registers": "ADC<k>"}``: which ADC each stream module (a CMACC) is fed by.

    The routing is set on the host (``Acadia._request_stream_configuration``), not by the
    sequencer program, so it is read from acadia: a configuration's ``input_switch_slave`` is
    the module number (``Acadia.cmacc_done`` polls ``module{input_switch_slave}_registers``)."""
    out = {}
    for cfg in getattr(acadia, "_stream_configurations", []) or []:
        source = getattr(cfg, "input_source", None)
        slave = getattr(cfg, "input_switch_slave", None)
        if slave is None or source is None or not hasattr(source, "num"):
            continue
        if getattr(source, "is_dac", False):
            continue
        out[f"module{slave}_registers"] = f"ADC{source.num()}"
    return out


def execute(program, bus, cache, choices=None, max_steps=50_000_000, cache_base=None,
            timing=None, streams=None, control=None):
    """Run ``program`` (from :func:`compile_program`) against one point's ``cache``
    ({word index: value}) and return the :class:`Execution`, with every instruction placed on
    the sequencer clock and every DMA descriptor on its channel's.

    ``choices``: {program address: bool} for branches on unknown values (taken or not).
    ``timing``: :func:`default_timing` (from the firmware's bus pipeline) unless given.
    ``streams``: :func:`stream_sources` -- which ADC's capture a CMACC poll waits for.
    ``control``: a :class:`ControlFlow` -- decides the runtime's test/loop branches (pins
    included) and records which pass of which construct every trigger belongs to.
    """
    streams = streams or {}
    choices = choices or {}
    timing = timing or default_timing()
    lag = timing["status_lag"]
    if cache_base is None:
        cache_base = next(start for start, _size, name in bus if name == "cache")
    regs = [0] * 8
    dsps = [DSPSlice() for _ in range(8)]
    mask = 0
    read_address = None        # address of the last bus READ (BUS_ADDR alone)
    stack = []
    run = Execution(timing=timing, control=control)
    pc, step, t = 0, 0, 0
    n = len(program)
    device_cache = {}
    channels = run.channels
    by_bit = {}
    capture_ends = []          # end of every capture window (ADC, not a dwell), for CMACC waits

    def channel(name):
        if name not in channels:
            channels[name] = DmaChannel(name, timing)
            by_bit[_channel_bit(name)] = channels[name]
        return channels[name]

    def device_of(address):
        hit = device_cache.get(address)
        if hit is None:
            hit = device_cache[address] = _device(bus, address)
        return hit

    def dma_name(name):
        return name[:3].upper() + name[3:-4] if name and name.endswith("_dma") and name[:3] in ("dac", "adc") else None

    def bus_read(at):
        """The word on BUS_DATA for the instruction at cycle ``at``."""
        if read_address is None:
            return None
        name, offset = device_of(read_address)
        if name == "cache":
            value = cache.get(offset)
            return None if value is None else int(value) & MASK32
        if name == "dma_running":
            word = 0
            for bit, ch in by_bit.items():
                if ch.running(at - lag):
                    word |= 1 << bit
            return word
        ch_name = dma_name(name)
        if ch_name is not None and offset == 0:
            return channel(ch_name).status_word(at - lag)
        run.unknown_reads[name] = run.unknown_reads.get(name, 0) + 1
        return None

    def source(kind, minor, imm, at):
        if kind == "IMM":
            return imm
        if kind == "REG":
            return regs[minor]
        if kind == "BUS_DATA":
            return bus_read(at)
        if kind == "DSP_P":
            d = dsps[minor]
            return None if d.counting or d.p is None else d.p & MASK32
        if kind == "STACK":
            if not stack:
                # acadia_sequencer.vhd: an empty stack still reads stack[0], whatever it holds
                run.problems.append(f"@{pc}: return with nothing pushed (stack underflow); the "
                                    f"board reads a stale stack entry, not modelled")
                return None
            return stack.pop()
        if kind == "PC":
            return pc
        run.problems.append(f"@{pc}: source {kind} not modelled")
        return None

    def alu(ins, value):
        """The sequencer ALU on src2: premask invert, then AND / XOR with MASK. It feeds both
        the condition and the OP source (``OP -> REG0 | BUS_DATA``, op=and: REG0 <- BUS_DATA &
        MASK -- how a CMACC result bit is stored)."""
        if value is None:
            return None
        if ins.premask_invert:
            value = ~value & MASK32
        if ins.op == "and":
            value &= mask
        elif ins.op == "xor":
            value ^= mask
        return value

    def clock(index):
        d = dsps[index]
        pcin = dsps[index - 1].p if index > 0 else 0
        d.p = dsp_output(d.config, d.ab, d.c, d.p, pcin)

    def holds_at(ins, h):
        value = alu(ins, bus_read(h))
        return None if value is None else (value != 0) != ins.condition_invert

    def release_of_hold(ins, h0):
        """First cycle >= h0 at which the hold lets go, or None if it never does."""
        name = device_of(read_address)[0] if read_address is not None else None
        if ins.s2 == "BUS_DATA" and name and re.fullmatch(r"module\d+_registers", name):
            # a CMACC "measurement complete" poll: released once the latest capture is in
            # a CMACC "measurement complete" poll: released once ITS ADC's latest capture window
            # has ended. (A raw-trace capture on another ADC must not hold it.)
            source = streams.get(name)
            ends = [p.stop for p in capture_ends if p.start is not None
                    and (source is None or p.channel == source)]
            if source is None:
                run.problems.append(f"@{pc}: CMACC poll on {name}, whose ADC is not known; "
                                    f"waited for the latest capture on any ADC")
            return max(h0, max(ends, default=h0) + timing["cmacc"])
        first = holds_at(ins, h0)
        if first is None:
            return "unknown"
        if not first:
            return h0
        if ins.s2 != "BUS_DATA" or not (name == "dma_running" or dma_name(name)):
            return None                 # a constant that holds: never released
        candidates = set()
        for ch in channels.values():
            candidates |= ch.change_times(h0 - lag)
        for tau in sorted(candidates):
            if not holds_at(ins, tau + lag):
                release = tau + lag
                if name == "dma_running" and one_cycle_join(mask):
                    release = max(h0, release - timing["one_cycle_join_early"])
                return release
        return None

    def one_cycle_join(bits):
        """Every channel the hold waits on last played ONE dwell of one cycle, starting exactly
        when its own trigger would start it (trigger cycle + ``trigger``) -- not held back behind
        anything still queued. The loopback releases such a join ONE_CYCLE_JOIN_EARLY cycles
        before the usual status-register wait (machine.py, measured 2026-10-08), also when a
        block=False pulse on the channel ends exactly there (join_after_short_batch.py,
        2026-11-18); a join queued behind a pulse or a descriptor left by an almost_empty drain
        is not early, and longer joins agree exactly."""
        waited = [ch for bit, ch in by_bit.items() if bits >> bit & 1]
        def joins(ch):
            d = ch.descriptors[-1] if ch.descriptors else None
            return (d is not None and d.ctype == 3 and d.length == 1 and d.start is not None
                    and ch.last_trigger is not None
                    and d.start == ch.last_trigger + timing["trigger"])
        return bool(waited) and all(joins(ch) for ch in waited)

    while step < max_steps:
        if not 0 <= pc < n:
            run.problems.append(f"PC ran off the program at {pc}")
            break
        if control is not None and pc in control.watch:
            skip_to = control.enter(pc)
            if skip_to is not None:          # an execution pinned to 0 passes: skip the loop
                control.last_branch = None
                pc = skip_to
                continue
        ins = program[pc]
        run.visits[pc] = run.visits.get(pc, 0) + 1
        next_pc = pc + 1
        step += 1

        if ins.conditional and ins.d1 == "PC" and ins.d1_minor in (
                Destination.PC_ABSOLUTE_HOLD, Destination.PC_RELATIVE_HOLD):
            # a PC HOLD: the instruction repeats until its condition stops holding
            release = release_of_hold(ins, t)
            if release == "unknown":
                name = device_of(read_address)[0] if read_address is not None else None
                run.problems.append(f"@{pc}: hold on an unknown value ({name})")
                break
            if release is None:
                run.problems.append(f"@{pc}: hold never releases")
                break
            if release > t:
                name = device_of(read_address)[0] if read_address is not None else None
                run.events.append(Event("wait", pc, step, device=name, mask=mask, time=t,
                                        end=release))
                # A hold that HELD wrote the PC with its source every cycle it held, so it
                # carries on from there (acadia_sequencer.vhd: a PC write keeps the instruction
                # register only for a hold). acadia makes that the next instruction, so this is
                # pc + 1 -- except where the compiler placed it otherwise (KI_004's out-of-line
                # test body, whose last hold resumes at the in-line return).
                if ins.s1 == "IMM":
                    next_pc = (ins.imm1 if ins.d1_minor == Destination.PC_ABSOLUTE_HOLD
                               else pc + _signed32(ins.imm1)) & 0xFFFF
                else:
                    run.problems.append(f"@{pc}: hold resumes at a non-immediate source "
                                        f"({ins.s1}); assumed the next instruction")
            t = release + 1
            pc = next_pc
            continue

        # src2 is read once: it feeds slot 2, the condition and the OP source alike
        v2 = source(ins.s2, ins.s2_minor, ins.imm2, t)
        v1 = alu(ins, v2) if ins.s1 == "OP" else source(ins.s1, ins.s1_minor, ins.imm1, t)

        if ins.conditional:
            # the condition: the ALU result of src2 (see alu) compared with zero
            test_value = alu(ins, v2)
            if control is not None and pc in control.by_branch:
                # a runtime test/loop: decided by ControlFlow (the value, a pin, or the default)
                known = None if test_value is None else (test_value != 0) != ins.condition_invert
                holds = control.decide(pc, known, (alu(ins, 0) != 0) != ins.condition_invert)
                if test_value is None:
                    run.assumed.setdefault(pc, []).append(holds)
                    if run.first_assumed_step is None:
                        run.first_assumed_step = step
            elif test_value is None:
                taken = choices.get(pc)
                if taken is None:
                    # provisional: the unknown read as 0 (a measurement read as g, say)
                    taken = (alu(ins, 0) != 0) != ins.condition_invert
                run.assumed.setdefault(pc, []).append(taken)
                if run.first_assumed_step is None:
                    run.first_assumed_step = step
                holds = taken
            else:
                holds = (test_value != 0) != ins.condition_invert
            writes = [(ins.d1, ins.d1_minor, v1)] if holds else []
        else:
            writes = [(ins.d1, ins.d1_minor, v1), (ins.d2, ins.d2_minor, v2)]

        # a bus WRITE is BUS_ADDR and BUS_DATA written by the same instruction
        dests = {w[0]: w for w in writes}
        if "BUS_ADDR" in dests and "BUS_DATA" in dests:
            address, data = dests["BUS_ADDR"][2], dests["BUS_DATA"][2]
            name, offset = device_of(address) if address is not None else (None, None)
            ch_name = dma_name(name)
            if ch_name is not None and offset in (0, 1, 2, 3):
                word = None if data is None else data & MASK32
                run.events.append(Event("dma", pc, step, channel=ch_name, ctype=offset,
                                        value=word, time=t))
                if word is None:
                    run.problems.append(f"@{pc}: {ch_name} command word unknown")
                played = Played(ch_name, offset, word, pc, t, t + timing["push"])
                channel(ch_name).push(played)
                if ch_name.startswith("ADC") and offset != 3:
                    capture_ends.append(played)
            elif ch_name is not None and offset == 4:
                channel(ch_name).reset()
                run.events.append(Event("write", pc, step, device=name, offset=offset,
                                        value=data, time=t))
            elif name == "dma_trigger":
                run.events.append(Event("trigger", pc, step, value=data, device=name, time=t,
                                        path=control.path(pc) if control is not None else ()))
                for bit in range(32):
                    if data is not None and data >> bit & 1:
                        name_bit = f"DAC{bit}" if bit < 16 else f"ADC{bit - 16}"
                        channel(name_bit).trigger(t, step)
            else:
                run.events.append(Event("write", pc, step, device=name, offset=offset,
                                        value=data, time=t))
            writes = [w for w in writes if w[0] not in ("BUS_ADDR", "BUS_DATA")]

        if ins.push_return:
            # acadia_sequencer.vhd pushes the return address whenever push_return is set,
            # whether or not the (conditional) branch is taken
            stack.append(pc + 1)
        halted = False
        branched = False
        configured = []
        for dest, minor, value in writes:
            if dest == "NONE":
                continue
            if dest == "REG":
                regs[minor] = None if value is None else value & MASK32
            elif dest == "MASK":
                mask = 0 if value is None else value & MASK32
                if value is None:
                    run.problems.append(f"@{pc}: MASK loaded from an unknown value")
            elif dest == "BUS_ADDR":
                read_address = value
            elif dest == "DSP_AB":
                dsps[minor].ab = value
            elif dest == "DSP_C":
                dsps[minor].c = value
            elif dest == "DSP_CFG":
                if value is None:
                    run.problems.append(f"@{pc}: DSP{minor} configured from an unknown value")
                    continue
                d = dsps[minor]
                d.config = decode_dsp_config(value)
                if d.config["rst_p"]:
                    d.p = 0
                d.counting = d.config["cep"] == 1
                if d.config["cep"] == 3:
                    configured.append(minor)        # one enabled clock after this write
            elif dest == "PC":
                if minor == Destination.PC_ABSOLUTE_BRANCH:
                    if value is None:
                        run.problems.append(f"@{pc}: branch to an unknown address")
                        halted = True
                    else:
                        next_pc = value
                        branched = True
                elif minor == Destination.PC_RELATIVE_BRANCH:
                    next_pc = pc + _signed32(value or 0)      # a 32-bit offset, may be negative
                    branched = True
                elif minor in (Destination.PC_RELATIVE_HOLD, Destination.PC_ABSOLUTE_HOLD):
                    halted = True                   # unconditional hold: halt()
            elif dest == "BUS_DATA":
                pass                                # a write without an address: no effect
            else:
                run.problems.append(f"@{pc}: destination {dest} not modelled")
        for minor in configured:
            clock(minor)
        if ins.cep is not None:
            clock(ins.cep)
        if halted:
            run.halted = True
            break
        t += 1 + (timing["branch"] if branched else 0)
        if control is not None:
            control.last_branch = pc if branched else None
        pc = next_pc
    else:
        run.problems.append(f"stopped after {max_steps} steps (no halt)")
        run.stop_pc = pc
    run.steps = step
    run.cycles = t
    return run


# ---------------------------------------------------------------------------------------------
# Words -> commands
# ---------------------------------------------------------------------------------------------

def decode_word(ctype, word):
    """``(kind, address, length)`` of a DMA command word (``Acadia.command_dma``):
    type 1 arbitrary = address << 16 | length - 1; types 0/2/3 = length - 1."""
    if word is None:
        return ({0: "ARB_CONT", 1: "ARB", 2: "CONST_CONT", 3: "DWELL"}.get(ctype), None, None)
    if ctype == 1:
        return "ARB", word >> 16, (word & 0xFFFF) + 1
    return {0: "ARB_CONT", 2: "CONST_CONT", 3: "DWELL"}.get(ctype), None, word + 1


# ---------------------------------------------------------------------------------------------
# Execution -> the trace's placements (what SeeQuence draws)
# ---------------------------------------------------------------------------------------------

def lay_out(trace, run):
    """Fill ``trace.placements`` from an :class:`Execution`: every descriptor the board plays, at
    the cycle the emulator's clock plays it. Times are shifted so the first sample is cycle 0.

    A placement is one executed TRIGGER and the descriptors it starts or that join its run; its
    block is the compiled block that trigger belongs to (``trace.control_flow["triggers"]``), so
    labels, conditionals and the block list stay the tracer's. Mutates and returns ``trace``.
    """
    from .tracing import Command, Placement

    played = [d for ds in run.played().values() for d in ds if d.start is not None]
    shift = min((d.start for d in played), default=0)
    # trigger program address -> block index
    block_of_trigger = dict(getattr(trace, "block_triggers", None) or {})
    if not block_of_trigger:            # an older trace: number triggers in address order
        nth = -1
        triggers = (trace.control_flow or {}).get("triggers", [])
        for index, block in enumerate(trace.blocks):
            if block.trigger:
                nth += 1
                if nth < len(triggers):
                    block_of_trigger[triggers[nth]] = index
    events = {e.step: e for e in run.events if e.kind == "trigger"}
    # a descriptor belongs to the first trigger of ITS channel executed after it was pushed --
    # the block it was pushed for -- even when it actually joins a run that is still playing
    trigger_steps = {}
    for e in events.values():
        for bit in range(32):
            if e.value is not None and e.value >> bit & 1:
                ch = f"DAC{bit}" if bit < 16 else f"ADC{bit - 16}"
                trigger_steps.setdefault(ch, []).append(e.step)
    step_of_push = {}
    for e in run.events:
        if e.kind == "dma":
            step_of_push[(e.channel, e.time)] = e.step

    groups = {}
    for d in played:
        push_step = step_of_push.get((d.channel, d.push_time), 0)
        owner = next((s for s in trigger_steps.get(d.channel, []) if s > push_step), d.trigger_step)
        groups.setdefault(owner, []).append(d)

    comments = {op.address: op.comment or "" for op in (trace.program or [])}
    # a DMA word written from BUS_DATA or a register is runtime data, not a compiled constant
    data_word = {op.address for op in (trace.program or []) if op.s2 in ("BUS_DATA", "REG", "DSP_P")}
    placements = []
    for step in sorted(groups, key=lambda s: (s is None, s)):
        event = events.get(step)
        index = block_of_trigger.get(event.address) if event is not None else None
        block = trace.blocks[index] if index is not None else None
        commands = []
        for d in sorted(groups[step], key=lambda d: (d.channel, d.start)):
            kind = KIND_NAMES[d.ctype]
            address = (d.word >> 16) if d.ctype == 1 and d.word is not None else None
            io_name = pulse = None
            num = (trace.dac_channel_nums or {}).get(d.channel)
            if address is not None and num is not None:
                io_name, pulse = (trace.addr_names or {}).get((num, address), (None, None))
            commands.append(Command(
                channel=d.channel, kind=kind, start=d.start - shift, length=d.length,
                # barrier padding: acadia builds its length as max(...) of the other branches
                is_padding=kind == "DWELL" and "function max" in comments.get(d.push_pc, ""),
                resolution="cache", pulse=pulse, io_name=io_name, address=address,
                streamed=d.push_pc in data_word))
        start = min(c.start for c in commands)
        stop = max(c.stop for c in commands)
        # which pass of every enclosing construct this trigger ran in (ControlFlow.path); the
        # innermost construct's pass is the placement's `iteration` (0 inside a test)
        path = tuple(event.path) if event is not None else ()
        placements.append(Placement(
            index=index if index is not None else -1,
            iteration=path[-1] if path else 0, path=path,
            start=start, length=stop - start,
            trigger=block.trigger if block else True, blocking=block.blocking if block else False,
            commands=commands, conditional=block.conditional if block else (),
            # everything triggered after a branch on a board-decided value rests on a guess
            assumed=(run.first_assumed_step is not None and step is not None
                     and step > run.first_assumed_step)))
    # The dead time after a blocking block: from its end to the first sample the NEXT trigger
    # can produce (that trigger's cycle + the trigger latency). Measured to the next block's first
    # sample instead, a block re-arming a channel that is still busy (a long raw capture) would
    # count that channel's playout as dead time. `placements` is in executed (trigger) order here.
    steps = sorted(groups, key=lambda s: (s is None, s))
    for k, (step_a, a) in enumerate(zip(steps, placements)):
        nxt = steps[k + 1] if k + 1 < len(steps) else None
        if a.blocking and nxt in events:
            a.gap_after = max(events[nxt].time + run.timing["trigger"] - shift - a.stop, 0)
    placements.sort(key=lambda p: p.start)
    trace.placements = placements
    trace.unresolved = 0
    control = getattr(run, "control", None)
    if control is not None:
        record_control_flow(trace, control, run)
    else:
        trace.assumed_paths = set(run.assumed)
    return trace


def record_control_flow(trace, control, run):
    """Write what the execution did at every construct into the fields the control-flow tabs
    and panel read (SequenceTrace.control_flow_summary, plotting.branch_regions):

    repeat_counts      {(first, level): passes} and {first: passes} for counts the DATA decided
                       (the first execution's; every execution's is in execution_counts)
    execution_counts   {(first, level, enclosing path): passes, or body ran? for a test}
    entered_paths      {(first, level): {enclosing path, ...}} -- every execution reached
    indeterminate      constructs decided on a value the board decides (a measurement)
    assumed_paths      their first blocks; assumed_constructs their keys (no pin in force)
    nonterminating     a loop still passing when the step limit stopped the run
    """
    trace.repeat_counts, trace.execution_counts, trace.entered_paths = {}, {}, {}
    # the older layout's unrolled-stream caption ("N gates from cache"); here a stream loop is an
    # ordinary repeat_until whose passes are counted like any other
    trace.stream_gates = trace.stream_words = 0
    trace.indeterminate, trace.nonterminating = set(), set()
    trace.assumed_paths, trace.assumed_constructs, trace.unsupported_paths = set(), set(), set()
    kinds = {(c.first, c.level): c.kind for c in control.by_branch.values()}
    for key, runs in control.executions.items():
        first, level = key
        trace.entered_paths[key] = set(runs)
        for outer, value in runs.items():
            trace.execution_counts[(first, level, outer)] = value
        if key in control.unknown:
            trace.indeterminate |= {key, first}
            if key not in control.pinned:
                trace.assumed_paths.add(first)
                trace.assumed_constructs.add(key)
        elif kinds.get(key) != "test" and runs and key not in control.pinned:
            count = int(next(iter(runs.values())))
            trace.repeat_counts[key] = count
            trace.repeat_counts.setdefault(first, count)
    # test(speculation=False): acadia puts the return in line and the body out of line, so
    # the executed path ends in a stack underflow (KI_004); keep the "not modelled" caption
    for c in control.by_branch.values():
        if c.kind == "test" and c.speculation is False:
            trace.unsupported_paths.add(c.first)
    # a construct this point never reached ran 0 times -- a fact of this execution, as long as
    # nothing around it was assumed (then the enclosing construct is marked assumed)
    for c in control.by_branch.values():
        key = (c.first, c.level)
        if key not in control.executions:
            trace.entered_paths.setdefault(key, set())
            # skipped because an enclosing arm was guessed (a measurement): not a fact, so it
            # keeps the board-decided default
            guessed = any((o.first, o.level) in control.unknown
                          and (o.first, o.level) not in control.pinned
                          for o in control.stack(c.branch) if o.level < c.level)
            if guessed:
                # whether (and how often) it runs is the board's decision: keep the "?" marker,
                # also once you pin it (gui_validation: XEB's active-reset loops, 2026-11-18)
                trace.indeterminate |= {key, c.first}
            elif c.kind != "test":
                trace.repeat_counts[key] = 0
                trace.repeat_counts.setdefault(c.first, 0)
    # Stopped by the step limit: the loop that never exits is the one still passing where the
    # run stopped -- of the loops around that pc, the one with the most passes (an inner loop
    # finishing normally restarts its count each execution). A loop records its execution only
    # when it exits, so this one has no count of its own: it is drawn x-infinity.
    if not run.halted and run.stop_pc is not None:
        loops = [c for c in control.stack(run.stop_pc) if c.kind != "test"]
        if loops:
            c = max(loops, key=lambda c: control.passes[c.id])
            trace.nonterminating |= {(c.first, c.level), c.first}


KIND_NAMES = {0: "ARB_CONT", 1: "ARB", 2: "CONST_CONT", 3: "DWELL"}
