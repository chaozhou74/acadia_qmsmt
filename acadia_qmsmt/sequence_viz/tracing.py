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
import linecache
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from acadia.compiler import ManagedResource, Operation
from acadia.system import DMASynchronizer
from acadia.sequencer import STP, Destination

from .dryrun import (BUS_ADDRESS_ATTR, BUS_STATE_ATTR, StopDryRun, already_traced, branch_recorder,
                     describe_operation, hardware_stubbed,
                     preserved_runtime_state)

KIND = {
    DMASynchronizer.ARBITRARY_CONTINUED: "ARB_CONT",
    DMASynchronizer.ARBITRARY: "ARB",
    DMASynchronizer.CONSTANT_CONTINUED: "CONST_CONT",
    DMASynchronizer.DWELL: "DWELL",
    DMASynchronizer.DIRECT: "DIRECT",
}

# The compiled program is read through structured STP fields (see `decode_program` and the
# `_is_*` predicates below), not by matching `instruction.pprint()` text.

# Sequencer and DMA timing, read off the gateware (acadia_sequencer.vhd, acadia_dma.vhd and the
# bus modules pyacadia generates), in sequencer cycles. The firmware-dependent parts are in
# `gateware_timing`.
#
# A taken branch costs this many cycles on top of its own: the PC register, the instruction
# memory's one-cycle read and the instruction register refill before the target executes.
TAKEN_BRANCH_CYCLES = 3
# A PC hold re-executes until its condition reads clear. If it held at least once, the next
# instruction executes this many cycles after that read (the fetch restarts, as after a branch,
# but the PC is already pointing past the hold). A hold whose condition is already clear the
# first time it executes falls through in one cycle like any other instruction.
HOLD_EXIT_CYCLES = 3
# A DMA loads a descriptor on one clock edge and registers its output on the next, so the first
# sample leaves this many cycles after the descriptor is pulled from the FIFO.
DMA_START = 2

# A counter-driven `repeat_until(counter == target)`: two operands (register/DSP names or
# literals) around a comparison. Used to resolve how many times such a loop runs.
COUNTER_RE = re.compile(r"^(\w+)\s*(==|!=|<=|>=|<|>)\s*(\w+)$")
# The incrementing counter in that idiom is a DSP (loaded 0, +1 per pass); the other operand
# is the target. `REG <op> literal` waits are NOT this -- their register is the varying side.
COUNTER_NAME_RE = re.compile(r"DSP\d+")

# A rendered condition split into its two operands and comparison -- deliberately loose on
# the operand side (each may itself be a compound expression like "REG3 & 524288") since the
# only thing this is used for is recognizing the complementary-pair idiom below.
GENERAL_CONDITION_RE = re.compile(r"^(.*?)\s*(==|!=|<=|>=|<|>)\s*(.+)$")
COMPLEMENT_OP = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


def _complement_text(condition_text):
    """The textual negation of a rendered condition (``"X != 0"`` -> ``"X == 0"``),
    or ``None`` if it doesn't parse as ``operand op operand``.

    Used to recognize the "two mutually-exclusive `test()` blocks" idiom this codebase
    uses to keep timing symmetric across a data-dependent branch (see
    ``SequenceTrace.execution_plan``) -- both conditions are usually indeterminate
    off-hardware (they gate on a live measurement), so the two look, textually, like
    exact negations of one another: ``"REG3 & 524288 != 0"`` / ``"REG3 & 524288 == 0"``.
    """
    match = GENERAL_CONDITION_RE.match((condition_text or "").strip())
    if not match:
        return None
    lhs, op, rhs = match.group(1).strip(), match.group(2), match.group(3).strip()
    return f"{lhs} {COMPLEMENT_OP[op]} {rhs}"


def _same_context(a, b):
    """Whether two block-level control-flow contexts are the SAME ``test``/``loop``/
    ``repeat_until`` occurrence, for grouping in :meth:`SequenceTrace._plan_level`.

    Every field compares by plain ``==`` except ``condition_obj`` and
    ``register_exprs``, which compare by identity. Two reasons: (1) an
    ``Operation``/``Register`` overloads comparison operators to build ANOTHER symbolic
    ``Operation`` rather than a bool (see :func:`_operation_value`), so ``==`` on one --
    or on a dict holding one -- raises when actually evaluated as a truth value;
    (2) identity is the semantically right test anyway -- every block compiled inside
    one ``with test(...):`` shares the exact same objects
    (``trace_runtime.spy_merge`` copies the context dict per block via ``dict(b)``,
    but a shallow copy keeps every value, including these, as the same object),
    while two genuinely different ``test()`` calls that happen to render identical
    text -- the same register reloaded and re-tested each pass of a Python loop, most
    commonly -- get fresh ones each time and so are correctly kept apart even though
    ``a["condition"] == b["condition"]``.
    """
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if key in ("condition_obj", "register_exprs"):
            if value is not b[key]:
                return False
        elif value != b[key]:
            return False
    return True


# A sequencer register/DSP as the rest of this module names one. Used to pick the
# register leaves out of a condition tree (anything else in there is a literal, a bus
# read or some other operand that is not a thing a user could pin).
REGISTER_NAME_RE = re.compile(r"^(?:REG|DSP)\d+$")


def _operation_leaves(node, out):
    """Collect the non-numeric leaf names of an ``Operation`` tree into ``out``."""
    if isinstance(node, Operation):
        for arg in node._args:
            _operation_leaves(arg, out)
    elif not isinstance(node, (int, np.integer, float, np.floating, bool)):
        out.add(str(node))


def _same_conditional(a, b):
    """Whether two blocks' full ``conditional`` tuples (outermost context first) are
    the same nesting path -- element-wise :func:`_same_context`, since tuple ``==``
    would otherwise hit the same non-bool-comparison problem one level down."""
    return len(a) == len(b) and all(_same_context(x, y) for x, y in zip(a, b))


def _operation_value(node, register_value=lambda name: None, bus_value=lambda address: None):
    """Evaluate a compiled condition/length expression to a concrete number, or
    ``None`` if some part of it isn't known.

    ``node`` is whatever the sequencer recorded for a ``test``/``repeat_until``
    condition or a command length: a plain number, an ``acadia.compiler.Operation``
    tree (built by applying Python operators to register/DSP handles -- see
    ``acadia.compiler.Operable``), or a bare Register/DSP handle. A leaf that isn't
    already numeric is looked up by its rendered name (``"REG0"``, ``"DSP2"``) via
    ``register_value``; the default never resolves one, so a tree of pure literals
    (acadia's own barrier-padding ``max()``/subtraction arithmetic, most commonly)
    folds to a concrete number while anything touching a real register does not.

    A ``BUS_DATA`` leaf tagged with its address by ``dryrun.branch_recorder`` (a cache
    read) goes to ``bus_value(node)`` instead.

    This calls the SAME operator object (``node._op``, e.g. ``operator.ne``) that
    produced the compiled instruction, rather than re-implementing comparison
    semantics -- consistent with the rest of this package hooking acadia's own
    machinery instead of re-modelling it.
    """
    if isinstance(node, Operation):
        args = [_operation_value(a, register_value, bus_value) for a in node._args]
        if any(a is None for a in args):
            return None
        try:
            return node._op(*args)
        except Exception:
            return None
    if isinstance(node, (int, np.integer, float, np.floating, bool)):
        return node
    if getattr(node, BUS_ADDRESS_ATTR, None) is not None:
        return bus_value(node)
    return register_value(str(node))


# Every ``Register()``/``DSP()`` call site in this codebase is a plain assignment --
# ``rounds_reg = a.sequencer().Register()``, never a tuple-unpack or an attribute target --
# so the variable a register is bound to can be read straight off its own source line.
# See ``_variable_name_spy`` below.
REGISTER_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\S")


def _runtime_source_file(runtime):
    """The file the runtime's own ``main()`` is written in, or None.

    This is where ``sequence()`` -- and so every register the EXPERIMENTER declared --
    lives, for a live runtime, an installed one, or one rebuilt from a data folder's
    saved ``runtime.py`` alike. Used by :func:`_variable_name_spy` to tell those apart
    from registers the libraries allocate for themselves.
    """
    try:
        return type(runtime).main.__code__.co_filename
    except Exception:
        return None


def _register_spy(names_out, exprs_out, source_file):
    """A stand-in for ``ManagedResource.__call__`` that records two things about every
    ``Register``/``DSP`` the dry run creates, keyed the way :func:`describe_registers`
    names them (``"REG0"``, ``"DSP2"``).

    **The Python variable it is assigned to** (into ``names_out``) -- see below.

    **The expression it was last loaded with** (into ``exprs_out``), by wrapping the
    register class's own ``load``. This is what makes a register loaded by *arithmetic*
    resolvable at all: ``describe_registers`` reads the compiled program and so only
    recognises a direct ``BUS_DATA -> REGn`` bus read, but

        flag.load(rounds_reg & (1 << (k - 1)))      # a cache-driven round count
        qubit_e.load(readout.get_measurement("quadrant") & a.CMACC_QUADRANT_2)

    land in a register through a DSP, so neither ``flag`` nor ``qubit_e`` appears
    there at all -- and every ``test`` gating on one was drawn as merely assumed.
    ``load`` is handed the un-evaluated ``Operation`` tree, so recording it keeps the
    whole chain down to its cache-fed or device-fed leaves (see
    :meth:`SequenceTrace._resolve_leaf`).

    ``exprs_out`` is LIVE and mutates as the sequence compiles -- a scratch register is
    reloaded with a different expression before each use (``flag`` carries a different
    round's mask at every pass). :func:`~.dryrun.branch_recorder` therefore snapshots it
    into each control-flow context as that context is entered, which is what ties a
    ``test`` to the expression its register actually held *there*. That is also why
    this is recorded here rather than recovered from the compiled program afterwards:
    the compiled program has lost which write precedes which test.

    Read off the *source line* of the call site (via the caller's frame and
    ``linecache``) rather than, say, walking the runtime's ``__dict__`` for an
    attribute holding this object -- a register is overwhelmingly a local, not
    `self.foo`, and the source line is unambiguous where a reverse-attribute-scan
    would have to guess among aliases.

    Only call sites in ``source_file`` -- the runtime's own ``main()``, see
    :func:`_runtime_source_file` -- are named. Registers and DSPs are also allocated
    INSIDE the libraries, and their local variable names are meaningless to an
    experimenter reading the plot: acadia grabs a scratch DSP to evaluate any register
    expression (``current_dsp = self.DSP()``, sequencer.py) and acadia_qmsmt's
    measurement helpers allocate their own (``reg = a.sequencer().Register()``,
    qmsmt.py), so naming by call site alone labelled unrelated registers
    ``current_dsp`` / ``reg``. Those keep their bare ``REGn``/``DSPn`` name.

    Declines silently for anything that isn't a simple assignment -- a tuple unpack,
    an attribute target, an inline expression -- rather than risk a misleading name,
    and never lets its own bookkeeping break a dry run.

    **A DSP walking the cache** (``configure(mode="P+1")``, then ``pulse_cep()``) has its
    steps counted in compile order, ``exprs_out[("ceps", name)]``, and each load records
    ``("loads", name)`` and the count at the load, ``("cep_at_load", name)``. The body of a
    loop compiles once, so these counters say where the pointer is on the FIRST pass; the
    execution plan adds what later passes step and removes what skipped bodies would have
    (see :meth:`SequenceTrace._dsp_steps`).
    """
    # ManagedResource's own type_name ("Register") is not the abbreviation the rest of
    # this module names registers with ("REG0", from Register.__str__ in acadia's
    # sequencer.py -- see describe_registers) -- map the two kinds this package cares
    # about to that abbreviation; anything else this metaclass also drives
    # (Instruction, ZDMA, the per-sequencer Sequence pool, ...) is left alone.
    ABBREVIATION = {"Register": "REG", "DSP": "DSP"}
    orig_call = ManagedResource.__call__
    patched_loads = {}      # register/DSP class -> {attribute: original}, for restoring
    modes = {}              # "DSP0" -> its last configured mode

    def patch_load(klass):
        """Wrap this register class's ``load`` to record what it is loaded with.

        ``load`` lives on the class ``Sequencer.__init__`` builds per sequencer (see
        ``reg_dct`` there), so there is no module-level class to patch ahead of time --
        it is wrapped the first time an instance of it is created, and restored by
        :func:`restore`.
        """
        original = klass.__dict__.get("load")
        if original is None or klass in patched_loads:
            return
        patched_loads[klass] = {"load": original}

        def load(reg_self, value, *a, **k):
            try:
                name = str(reg_self)
                exprs_out[name] = value
                exprs_out[("loads", name)] = exprs_out.get(("loads", name), 0) + 1
                exprs_out[("cep_at_load", name)] = exprs_out.get(("ceps", name), 0)
            except Exception:      # never let bookkeeping break a dry run
                pass
            return original(reg_self, value, *a, **k)

        klass.load = load

        original_configure = klass.__dict__.get("configure")
        original_cep = klass.__dict__.get("pulse_cep")
        if original_configure is None or original_cep is None:
            return
        patched_loads[klass].update(configure=original_configure, pulse_cep=original_cep)

        def configure(dsp_self, *a, **k):
            try:
                if "mode" in k:
                    modes[str(dsp_self)] = str(getattr(k["mode"], "name", k["mode"]))
            except Exception:
                pass
            return original_configure(dsp_self, *a, **k)

        def pulse_cep(dsp_self, *a, **k):
            try:
                name = str(dsp_self)
                if modes.get(name) == "P+1":
                    exprs_out[("ceps", name)] = exprs_out.get(("ceps", name), 0) + 1
            except Exception:
                pass
            return original_cep(dsp_self, *a, **k)

        klass.configure = configure
        klass.pulse_cep = pulse_cep

    def spy(type_self, *args, **kwargs):
        instance = orig_call(type_self, *args, **kwargs)
        try:
            prefix = ABBREVIATION.get(getattr(type_self, "__name__", ""))
            if prefix is None:
                return instance
            patch_load(type_self)
            if source_file is None:
                return instance
            frame = sys._getframe(1)
            if frame.f_code.co_filename == source_file:
                line = linecache.getline(source_file, frame.f_lineno)
                match = REGISTER_ASSIGN_RE.match(line)
                if match:
                    names_out[f"{prefix}{instance._resource_id}"] = match.group(1)
        except Exception:      # never let naming break a dry run
            pass
        return instance

    def restore():
        for klass, originals in patched_loads.items():
            for attribute, original in originals.items():
                setattr(klass, attribute, original)
        patched_loads.clear()

    return orig_call, spy, restore


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
    # A symbolic length that happens to be pure literal arithmetic (acadia's own
    # barrier-padding max()/subtraction over already-known pulse lengths, typically)
    # folds to a concrete number here -- but ONLY for decluttering the register panel
    # (see SequenceTrace.register_summary). It is deliberately NEVER promoted to
    # `length`/a non-symbolic command: the schedule this is read from can carry a
    # padding entry that acadia's compiler elides rather than emitting as its own DMA
    # instruction, so trusting this value for layout invents commands that do not
    # exist in the archived compiled.log (confirmed against archived runs -- see
    # docs/DEVELOPER_NOTES.md).
    static_length: Optional[int] = None
    # a schedule_direct command: the whole DMA word comes from `symbolic` (a register)
    direct: bool = False
    # a register length issued as length - 1 (acadia's length_is_minus_one): it plays one more
    length_offset: int = 0

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
    register_exprs: dict = field(default_factory=dict)  # register loads as of this block

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
    path: tuple = ()            # pass index at each nesting level of ``conditional`` (0 for a test)

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
    written_pulses: set = field(default_factory=set)      # (io, pulse) load_pulse wrote
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
    loop_counts: dict = field(default_factory=dict)  # block index (or (block, depth)) -> iterations to draw
    repeat_counts: dict = field(default_factory=dict)  # block index -> resolved repeat_until count
    path_choices: dict = field(default_factory=dict)  # block index (or (block, depth)) -> take test body?
    # (first block, depth) -> what the last execution_plan drew for that loop / test and why
    # (see _resolve_context); read by control_flow_summary
    flow_state: dict = field(default_factory=dict)
    laid_out: bool = False           # relayout has run: an empty `placements` then means nothing runs
    assumed_paths: set = field(default_factory=set)   # tests we could not decide
    # of those, the ones that are one side of a complementary `test() `/`test()` pair
    # (see execution_plan) -- drawn taken, but arbitrarily so, not "probably taken"
    mutually_exclusive_paths: set = field(default_factory=set)
    unsupported_paths: set = field(default_factory=set)  # KI_004: speculation=False
    dma_timing: dict = field(default_factory=dict)  # see gateware_timing
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
    cache_base: Optional[int] = None                 # bus address of cache word 0
    channel_nums: dict = field(default_factory=dict)  # "DAC2" -> 2, for addr_names
    # per execution_plan entry, the steps each pointer DSP has taken beyond the compile
    # order: {"DSP0": (its load count, extra steps or None if unknown)}; see _dsp_steps
    plan_pointers: list = field(default_factory=list)
    plan_paths: list = field(default_factory=list)     # Placement.path of each plan entry
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
            self.written_pulses = set(snapshot.get("written") or ())
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
        self.laid_out = True
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
        return [c for b in self.drawn_blocks for c in b.commands]

    @property
    def drawn_blocks(self):
        """What executes: the placements once laid out -- empty when nothing runs, e.g. both
        sides of a test pair pinned off -- else, before the first layout, the compiled blocks."""
        return self.placements if self.laid_out else self.blocks

    @property
    def static_commands(self):
        """Commands as COMPILED, each appearing once. This is what ``compiled.log``
        contains, so cross-checks must use it rather than the unrolled view."""
        return [c for b in self.blocks for c in b.commands]

    def register_value(self, name):
        """This point's value for register/DSP ``name``, or None if not known.

        An explicit :attr:`register_overrides` entry wins over the per-point cache,
        so pinning a register is the one lever that works for a value the hardware
        only produces at run time (a measurement result) -- which is what makes a
        branch on it previewable: see :meth:`evaluate_condition`.
        """
        if name in self.register_overrides:
            return int(self.register_overrides[name])
        value = self.register_cycles.get(name)
        return int(value) if value is not None else None

    def _resolve_leaf(self, name, exprs, seen=frozenset()):
        """A register/DSP leaf's value, following what it was loaded WITH if needed.

        The expression the register was last loaded with before the ``test`` being
        evaluated (``exprs``, snapshotted per control-flow context by
        ``dryrun.branch_recorder``) is tried FIRST, resolved recursively down to ITS
        leaves. That is what carries a cache value through arithmetic:
        ``flag = rounds_reg & mask`` resolves because ``rounds_reg`` is cache-fed, so
        the round-enable tests of a runtime whose round count comes from the cache
        follow the sweep point instead of being assumed.

        Only when that fails does :meth:`register_value` apply -- an explicit override,
        else this point's cache. The order matters because a scratch register gets
        REUSED for unrelated things: ``flag`` may carry ``rounds_reg & mask`` at a
        round-enable test and ``a_g & b_g`` (two measurement results) at a feedback test a
        few lines later. Resolving the
        expression first means a pin on ``flag`` decides the branch that genuinely
        cannot be known, while the rounds keep resolving from the cache -- where letting
        the override win everywhere would force every round to the pinned value too.

        ``seen`` breaks a cycle (a register loaded from itself, e.g. an accumulator),
        which would otherwise recurse forever.
        """
        if name in seen:
            return None
        expression = (exprs or {}).get(name)
        if expression is not None:
            value = _operation_value(
                expression,
                lambda leaf: self._resolve_leaf(leaf, exprs, seen | {name}),
                lambda node: self._bus_value(node, exprs, seen | {name}))
            if value is not None:
                steps = self._dsp_steps(name, exprs)
                return None if steps is None else value + steps
        return self.register_value(name)

    def _bus_value(self, node, exprs, seen=frozenset()):
        """This point's cache word at a bus read -- its address a literal, or a DSP pointer
        resolved as it stood AT the read -- or None outside the cache / when unknown."""
        exprs = {**(exprs or {}), **(getattr(node, BUS_STATE_ATTR, None) or {})}
        value = _operation_value(getattr(node, BUS_ADDRESS_ATTR),
                                 lambda leaf: self._resolve_leaf(leaf, exprs, seen),
                                 lambda inner: self._bus_value(inner, exprs, seen))
        if value is None or self.cache_base is None:
            return None
        return (self.point_cache or {}).get(int(value) - self.cache_base)

    def _dsp_steps(self, name, exprs):
        """``pulse_cep`` steps DSP ``name`` has taken since its load, as of the moment
        ``exprs`` was snapshotted: the compile-order count plus what the execution plan has
        added (later loop passes) or removed (skipped bodies). None when a loop that steps
        it ran an unknown number of passes."""
        exprs = exprs or {}
        compiled = exprs.get(("ceps", name), 0) - exprs.get(("cep_at_load", name), 0)
        load, extra = getattr(self, "_pointer_extra", {}).get(name, (None, 0))
        if load != exprs.get(("loads", name)):
            extra = 0                    # reloaded since that extra was counted
        return None if extra is None else compiled + extra

    def _body_steps(self, context):
        """``{DSP: pulse_cep steps}`` one pass of this context's body takes (nested bodies
        counted once each), for the DSPs it steps and does not reload."""
        entry, exit_ = context.get("register_exprs") or {}, context.get("exit_counters") or {}
        steps = {}
        for key, value in exit_.items():
            if key[0] != "ceps":
                continue
            name = key[1]
            if exit_.get(("loads", name)) != entry.get(("loads", name)):
                continue                 # reloaded inside: each pass starts afresh
            step = value - entry.get(key, 0)
            if step:
                steps[name] = step
        return steps

    def _shift_pointers(self, context, passes):
        """Add ``passes`` x one body's steps to the pointers it steps (None: unknown). A DSP
        the body reloads (an inner loop's counter) starts every pass afresh, and after a
        skipped body its value is not the one the compiled order implies."""
        entry, loads = context.get("register_exprs") or {}, context.get("exit_counters") or {}
        for key, load in loads.items():
            if key[0] == "loads" and load != entry.get(key):
                self._pointer_extra[key[1]] = (load, None if passes is None or passes < 0 else 0)
        for name, step in self._body_steps(context).items():
            load = loads.get(("loads", name))
            held, extra = self._pointer_extra.get(name, (load, 0))
            if held != load:
                extra = 0
            self._pointer_extra[name] = (
                load, None if passes is None or extra is None else extra + passes * step)

    def _pointer_loop_count(self, context):
        """Passes of ``repeat_until(pointer == end)``: one pass steps the pointer by its
        body's ``pulse_cep`` count, so the count is how far ``end`` is from where the pointer
        stands at entry, in those steps -- whatever else the body reads in between (a gate
        string: one word per pass; several modes stepped together: one word per mode per
        pass). None when the
        condition is not that shape or the arithmetic is not a whole count."""
        condition = context.get("condition_obj")
        steps = self._body_steps(context)
        if (not isinstance(condition, Operation) or len(condition._args) != 2
                or getattr(condition._op, "__name__", "") != "eq"):
            return None
        exprs = context.get("register_exprs")
        for counter, target in (condition._args, condition._args[::-1]):
            step = steps.get(str(counter))
            if not step or not COUNTER_NAME_RE.fullmatch(str(counter)):
                continue
            now = self._resolve_leaf(str(counter), exprs)
            end = _operation_value(target, lambda leaf: self._resolve_leaf(leaf, exprs),
                                   lambda node: self._bus_value(node, exprs))
            if now is None or end is None:
                return None
            distance = int(end) - int(now)
            # do-while: a loop entered at its end never exits, so 0 is not a count
            if distance <= 0 or distance % step:
                return None
            return distance // step
        return None

    def evaluate_condition(self, condition_obj, register_exprs=None):
        """Try to decide a ``test`` condition from the register values of this point.

        Walks the compiled condition's ``Operation``/``Register`` tree (see
        :func:`_operation_value`), resolving each register/DSP leaf with
        :meth:`_resolve_leaf`: an explicit override, else this point's cache, else the
        expression that register was last loaded with -- recursively. So this resolves
        any expression over those, not just a bare ``REGn <op> literal``; a masked
        condition like ``(flag := rounds_reg & 1) != 0`` resolves from the cache word
        behind ``rounds_reg``.

        Pinning a measurement-fed register (``register_overrides["DSP0"] = 0``, or the
        GUI's register panel) therefore decides its branch outright -- the drawn arm is
        reported as resolved rather than assumed, and its complement in a
        mutually-exclusive pair follows automatically. Returns ``None`` when some leaf
        is still unknown, in which case the caller falls back to ``path_choices`` or to
        the mutually-exclusive/assumed default (see :meth:`execution_plan`).
        """
        if condition_obj is None:
            return None
        value = _operation_value(
            condition_obj, lambda leaf: self._resolve_leaf(leaf, register_exprs))
        if isinstance(value, (int, np.integer, float, np.floating, bool)):
            return bool(value)
        return None

    def _operand_value(self, token):
        """Resolve a condition operand to an int, or None.

        A literal, or a register pinned by an override / fed from this point's cache.
        Counters (DSPs) and anything unknown return None.
        """
        try:
            return int(token, 0)
        except ValueError:
            pass
        return self.register_value(token)

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
        count = self._operand_value(right if left_dsp else left)
        if count is None or count < 0:
            return None
        return count

    def execution_plan(self):
        """``[(block index, iteration), ...]`` in the order the sequencer runs them.

        Resolved level by level, outermost first: a block only executes if EVERY
        context that encloses it is taken, not just the innermost one -- a ``test``
        nested inside another ``test`` (an outer round-enable check wrapping an inner
        feedforward branch) only
        draws its body when both hold. A ``loop`` repeats its own deterministic count.
        A ``repeat_until`` repeats the count :meth:`repeat_until_count` resolves from
        its condition register/literal (recorded in :attr:`repeat_counts`); when that
        can't be resolved the count is data-dependent, so one pass is drawn. A user
        ``loop_counts[first block]`` overrides either; ``loop_counts[(first block, depth)]``
        pins one construct only, where a loop nested in another starts at the same block.

        A ``test`` whose condition can't be decided from the cache falls back to
        :attr:`path_choices`, then to an arbitrary default -- except this codebase's
        idiom for keeping a data-dependent branch's timing symmetric is to compile
        it as TWO mutually-exclusive ``test()`` blocks (``REGn != 0`` / ``REGn == 0``
        on the same register, so exactly one runs on hardware): defaulting each
        independently draws both as taken, which is the "always shows the true
        outcome twice" complaint this replaces. Instead, every ``test`` in the trace
        whose condition has a counterpart that is its exact textual negation
        (:func:`_complement_text`) picks a side once and every later block testing
        that same condition -- or its negation -- mirrors it, so the pair (or a
        register reused across several such pairs, as in a feedforward repeated over
        rounds) reads as one consistent, single-valued outcome
        rather than as if every arm ran. Force either side without re-tracing via
        ``trace.path_choices[block_index] = False``; :attr:`mutually_exclusive_paths`
        lists which blocks this applied to.
        """
        self.repeat_counts = {}
        self.assumed_paths = set()
        self.unsupported_paths = set()
        self.mutually_exclusive_paths = set()
        self.flow_state = {}
        self._condition_decisions = {}
        self._pinned_conditions = self._find_pinned_conditions()
        self._paired_conditions = self._find_paired_conditions()
        self._pointer_extra = {}
        self.plan_pointers = []
        self.plan_paths, self._path = [], []
        plan = self._plan_level(list(range(len(self.blocks))), 0, 0)
        self._pointer_extra = {}
        return plan

    def _find_pinned_conditions(self):
        """``{condition text: taken}`` for the tests pinned by a ``(block, depth)`` key of
        :attr:`path_choices`. A test whose exact complement is pinned follows that pin, ahead of
        its own cache value: of a complementary pair exactly one side runs, so pinning one side
        decides both."""
        pinned = {}
        for key, taken in self.path_choices.items():
            if not (isinstance(key, tuple) and len(key) == 2 and taken is not None):
                continue
            index, depth = key
            if 0 <= index < len(self.blocks) and depth < len(self.blocks[index].conditional):
                context = self.blocks[index].conditional[depth]
                if context.get("kind") == "test" and context.get("condition") is not None:
                    pinned[context["condition"]] = bool(taken)
        return pinned

    def _find_paired_conditions(self):
        """Rendered ``test`` condition texts that have their exact negation
        (:func:`_complement_text`) appearing SOMEWHERE else in this trace -- at any
        nesting depth, since a register (and so its condition text) can be reused
        across separate ``test()`` pairs (once on its own, once as the outer half of a
        nested pair). Membership here is what :meth:`_plan_level` uses to caption a block
        "mutually exclusive" rather than a bare "assumed taken".
        """
        texts = {ctx["condition"] for block in self.blocks for ctx in block.conditional
                 if ctx["kind"] == "test"}
        return {t for t in texts if _complement_text(t) in texts}

    def _plan_level(self, indices, depth, iteration):
        """Resolve one nesting level and recurse into each body's children.

        ``indices`` all share an identical ``conditional[:depth]`` prefix (the whole
        list, on the initial call); grouped here by ``conditional[depth]`` -- ``None``
        for an index whose tuple ends at this depth, i.e. it sits directly in the
        parent body with no further nesting of its own.
        """
        groups, i = [], 0
        while i < len(indices):
            block = self.blocks[indices[i]]
            context = block.conditional[depth] if depth < len(block.conditional) else None
            j = i + 1
            if context is not None:
                while (j < len(indices)
                       and depth < len(self.blocks[indices[j]].conditional)
                       and _same_context(self.blocks[indices[j]].conditional[depth],
                                         context)):
                    j += 1
            groups.append((context, indices[i:j]))
            i = j

        plan = []
        for context, body in groups:
            if context is None:
                for index in body:
                    plan.append((index, iteration))
                    # entries are made in execution order, so this lines up with the plan
                    self.plan_pointers.append(dict(self._pointer_extra))
                    self.plan_paths.append(tuple(self._path))
            else:
                self._path.append(0)              # a loop's passes set it in _passes
                plan.extend(self._resolve_context(body, context, depth, iteration))
                self._path.pop()
        return plan

    @staticmethod
    def _flow_override(overrides, first_index, depth):
        """A user pin from ``loop_counts`` / ``path_choices`` for the construct whose body starts at
        ``first_index`` at nesting ``depth``: keyed ``(block, depth)``, else by the block alone."""
        key = (first_index, depth)
        return overrides[key] if key in overrides else overrides.get(first_index)

    def _record_flow(self, first_index, depth, **state):
        """Note what was drawn for a construct. A construct inside a loop is resolved once per
        pass: the first pass gives the source (later passes of an undecided test only mirror
        it), and a test outcome or pass count that differs between passes is recorded as None."""
        key = (first_index, depth)
        old = self.flow_state.get(key)
        if old is not None:
            state["source"] = old["source"]
            for name in ("taken", "count"):
                if name in state and old.get(name) != state[name]:
                    state[name] = None
        self.flow_state[key] = state

    def _resolve_context(self, body, context, depth, iteration):
        first_index = body[0]
        kind = context["kind"]

        if kind == "test" and context.get("speculation") is False:
            # KI_004: with speculation=False the body is placed OUT OF LINE, so address
            # order stops being execution order and the edge costing below does not
            # apply -- measured 25 ns out on the taken arm, and the skipped arm hangs
            # the sequencer outright. Draw the body but flag the path as unmodelled
            # rather than assert a timeline we know is wrong.
            self.assumed_paths.add(first_index)
            self.unsupported_paths.add(first_index)
            self._record_flow(first_index, depth, taken=True, source="unsupported")
            return self._plan_level(body, depth + 1, iteration)

        if kind == "test":
            condition_text = context.get("condition")
            # explicit choice wins; otherwise try to decide it from this point's cache
            taken = self._flow_override(self.path_choices, first_index, depth)
            source = "pinned"
            if taken is None:
                # the other side of a pinned complementary pair runs exactly when the pinned
                # side does not, whatever this point's cache says
                complement = _complement_text(condition_text)
                if complement is not None and complement in self._pinned_conditions:
                    taken, source = not self._pinned_conditions[complement], "follows"
            if taken is None:
                taken = self.evaluate_condition(context.get("condition_obj"),
                                                context.get("register_exprs"))
                source = "cache"
            if taken is None:
                # neither -- fall back to the mutually-exclusive-pair default (or a bare
                # arbitrary default if this condition has no known complement)
                source = "follows"
                if condition_text in self._condition_decisions:
                    taken = self._condition_decisions[condition_text]
                else:
                    complement = _complement_text(condition_text)
                    if complement is not None and complement in self._condition_decisions:
                        taken = not self._condition_decisions[complement]
                    else:
                        taken, source = True, "assumed"
                self.assumed_paths.add(first_index)
                if condition_text in self._paired_conditions:
                    self.mutually_exclusive_paths.add(first_index)
            # record whichever way this was decided -- including an explicit
            # path_choices override -- so a later occurrence of this same physical
            # condition, or its exact complement, stays consistent with it
            if condition_text is not None:
                self._condition_decisions[condition_text] = taken
            self._record_flow(first_index, depth, taken=bool(taken), source=source)
            if not taken:
                self._shift_pointers(context, -1)    # its steps were compiled, never run
                return []
            return self._plan_level(body, depth + 1, iteration)

        if kind == "repeat_until":
            # count from the loop's condition register/literal when resolvable
            # (recorded so the caption can state it); else data-dependent -> one pass.
            count = self._flow_override(self.loop_counts, first_index, depth)
            source = "pinned"
            # a cache-pointer stream unrolls its whole train inside one pass (_expand_stream)
            streamed = any(self._is_stream_command(c) for i in body for c in self.blocks[i].commands)
            if count is None and not streamed:
                source = "cache"
                count = self._pointer_loop_count(context)
                if count is None:
                    count = self.repeat_until_count(context)
                if count is not None:
                    self.repeat_counts[first_index] = int(count)
            if count is None:
                source = "stream" if streamed else "assumed"
            self._record_flow(first_index, depth, count=1 if count is None else int(count), source=source)
            known = count is not None
            plan = self._passes(body, context, depth, 1 if count is None else int(count))
            if not known:
                self._shift_pointers(context, None)
            return plan

        # loop: deterministic count, unrolled.
        count = self._flow_override(self.loop_counts, first_index, depth)
        source = "pinned"
        if count is None:
            count, source = context.get("count") or 1, "compiled"
        self._record_flow(first_index, depth, count=max(int(count), 1), source=source)
        return self._passes(body, context, depth, max(int(count), 1))

    def _passes(self, body, context, depth, count):
        """``count`` passes of a loop body. The body compiled once, so every pass after the
        first steps its pointers one body further than the compile-order counters say."""
        plan = []
        for n in range(max(count, 0)):
            if n:
                self._shift_pointers(context, 1)
            self._path[-1] = n
            plan.extend(self._plan_level(body, depth + 1, n))
        if count <= 0:
            self._shift_pointers(context, -1)
        return plan

    @property
    def length_cycles(self):
        return max((b.stop for b in self.drawn_blocks), default=0)

    @property
    def length_ns(self):
        return self.length_cycles * self.ns_per_cycle

    def register_label(self, name):
        """``REG0`` plus what it is fed from -- ``t_echo = cache[0]`` if aliased."""
        alias = self.register_names.get(name, name)
        source = self.registers.get(name, {}).get("source")
        return f"{alias} = {source}" if source else alias

    def condition_registers(self):
        """Register/DSP names that gate a ``test`` -- the ones worth pinning.

        Taken from the condition trees themselves rather than from
        :meth:`describe_registers`, because the register a branch actually turns on is
        often invisible to that: a ``qubit_e`` loaded from a DSP holding
        ``measurement & QUADRANT_2`` so the compiled program shows only the
        scratch DSP and the bus read, never a ``BUS_DATA -> REG0`` to attribute to it.
        The condition names it directly.

        :return: ``{name: True only if it resolves at EVERY test it gates}``. Aggregated
            that way because a scratch register is reused: a ``flag`` may resolve at a
            round-enable test
            (``rounds_reg & mask``) but not at the reload test (two measurement
            results), and it is the unresolvable use that needs a control.
        """
        found = {}
        for block in self.blocks:
            for context in block.conditional:
                if context.get("kind") != "test":
                    continue
                leaves = set()
                _operation_leaves(context.get("condition_obj"), leaves)
                for leaf in leaves:
                    if not REGISTER_NAME_RE.match(leaf):
                        continue
                    resolved = self._resolve_leaf(
                        leaf, context.get("register_exprs")) is not None
                    found[leaf] = found.get(leaf, True) and resolved
        return found

    def control_flow_summary(self):
        """Every ``loop`` / ``repeat_until`` / ``test`` of the sequence, for a UI that pins them.

        One entry per construct in program order, outermost first, grouped exactly as
        :meth:`execution_plan` groups them::

            {key, tag, kind, depth, label, condition, compiled_count, count, taken, source,
             reached, settable, paired}

        ``key`` is ``(first block, depth)``: write it into :attr:`loop_counts` (a loop's passes)
        or :attr:`path_choices` (whether a test's body runs), then :meth:`relayout`. ``count`` /
        ``taken`` / ``source`` are what the last layout drew and why -- ``pinned``, ``compiled``
        (a loop's own count), ``cache`` (decided from this point's registers), ``follows`` (a
        test mirroring an earlier test of the same register, see :meth:`execution_plan`),
        ``assumed`` (undecidable; a default was drawn), ``stream`` (a cache-pointer stream, one
        pass holding the whole train), ``unsupported`` (``speculation=False``). ``reached`` is
        False inside a branch that is not drawn. ``settable`` is False where a pin is ignored:
        a streamed loop and a ``speculation=False`` test. ``tag`` is a short name for the
        construct, ``@<first block>``, with ``.<level>`` added where two constructs begin at the
        same block -- the name its tab on the drawing carries.
        """
        paired = getattr(self, "_paired_conditions", set())
        entries = []

        def walk(indices, depth):
            i = 0
            while i < len(indices):
                context = (self.blocks[indices[i]].conditional[depth]
                           if depth < len(self.blocks[indices[i]].conditional) else None)
                j = i + 1
                if context is not None:
                    while (j < len(indices)
                           and depth < len(self.blocks[indices[j]].conditional)
                           and _same_context(self.blocks[indices[j]].conditional[depth], context)):
                        j += 1
                    body = indices[i:j]
                    entries.append(flow_entry(body, depth, context))
                    walk(body, depth + 1)
                i = j

        def flow_entry(body, depth, context):
            kind, condition = context.get("kind"), context.get("condition")
            state = self.flow_state.get((body[0], depth))
            if kind == "test":
                label = f"test {condition or ''}".strip()
                settable = context.get("speculation") is not False
            elif kind == "repeat_until":
                label = f"repeat until {condition or ''}".strip()
                settable = not any(self._is_stream_command(c) for i in body
                                   for c in self.blocks[i].commands)
            else:
                label = f"loop x{context.get('count') or 1}"
                settable = True
            return {"key": (body[0], depth), "kind": kind, "depth": depth, "label": label,
                    "condition": condition, "compiled_count": context.get("count"),
                    "count": (state or {}).get("count"), "taken": (state or {}).get("taken"),
                    "source": (state or {}).get("source"), "reached": state is not None,
                    "settable": settable, "paired": kind == "test" and condition in paired}

        walk(list(range(len(self.blocks))), 0)
        starts = {}
        for entry in entries:
            starts[entry["key"][0]] = starts.get(entry["key"][0], 0) + 1
        for entry in entries:
            block, depth = entry["key"]
            entry["tag"] = f"@{block}" if starts[block] == 1 else f"@{block}.{depth + 1}"
        return entries

    def register_summary(self):
        """Describe every register / length-symbol for a per-register UI control.

        Three kinds appear:

        * **cache-fed** registers (``resolution`` "cache") resolve themselves from the
          per-point cache, so their value is shown but not settable -- it is already
          right, and follows the sweep point. One may drive a command length
          (``is_length``, in cycles) or only a ``test``/``repeat_until`` condition
          (the raw register value -- which, when it is a loop target, is the count the
          loop is drawn for).
        * **device-fed** registers (``resolution`` "device") come from live hardware --
          a CMACC measurement result -- so they have no value off-hardware. Settable:
          pinning one decides the ``test`` it gates (see :meth:`evaluate_condition`),
          which is the only way to preview a specific arm of a feedforward branch.
        * **register/DSP-driven command lengths** not recoverable from the cache
          (``resolution`` "fallback"/"override"). Settable, in cycles.

        Everything settable is set through :attr:`register_overrides`; note the units
        differ by kind -- a length override is in cycles, while a device-fed register's
        is the raw register word the condition compares against (0 vs the bit the
        runtime masks for, typically), so ``value_cycles`` is only literally cycles
        when ``is_length``.

        :return: ``[{name, alias, label, source, resolution, value_cycles, is_length,
            settable}, ...]``, deduped by name, identified registers first.
        """
        # a symbolic length that folded to a plain number (Command.static_length) is
        # not a register at all -- just acadia's own padding arithmetic over already-
        # known pulse lengths -- so it does not belong in a REGISTER control panel
        resolved = {c.symbolic: (c.resolution, c.length)
                    for c in self.commands if c.symbolic and c.static_length is None}

        def entry(name, resolution, cycles, is_length):
            return {"name": name,
                    # "alias" is just the display name (the runtime's own variable
                    # name, if one was found -- see _variable_name_spy -- else the
                    # bare "REG0"/"DSP0"); "label" adds "= source" for a caption/
                    # tooltip. A UI row header wants the short one.
                    "alias": self.register_names.get(name, name),
                    "label": self.register_label(name),
                    "source": self.registers.get(name, {}).get("source") or "",
                    "resolution": resolution, "value_cycles": cycles,
                    "is_length": is_length,
                    # "cache" is the one kind that is NOT settable: it already carries
                    # this point's real value, and overriding it would just desync the
                    # picture from the sweep point being shown.
                    "settable": resolution in ("override", "fallback", "device")}

        gating = self.condition_registers()

        entries = {}
        for name, info in self.registers.items():             # cache/device registers
            if name in resolved:                              # also drives a length
                entries[name] = entry(name, *resolved[name], True)
            elif info["cache_word"] is not None:              # cache-fed
                entries[name] = entry(name, "cache",
                                      self.register_value(name), False)
            elif name in gating:                              # device-fed, gates a test
                # register_value, not register_cycles: a device register has no cache
                # value, so its displayed value is whatever the user pinned it to
                entries[name] = entry(name, "device",
                                      self.register_value(name), False)
            # a device register that neither drives a length nor gates a test is
            # acadia's own scratch (the DSP it grabs to evaluate a register
            # expression) -- nothing an experimenter can act on, so it is left out
        for name, (resolution, cycles) in resolved.items():   # DSP-driven lengths
            if name not in entries:
                entries[name] = entry(name, resolution, cycles, True)
        for name, resolves in gating.items():                 # registers gating a test
            # one that already resolves per point needs no control (and offering to
            # pin it would just invite desyncing the picture from the sweep point)
            if name not in entries and not resolves:
                entries[name] = entry(name, "device", self.register_value(name), False)
        return list(entries.values())

    def envelope(self, io_name, pulse, source="memory"):
        """Complex waveform for a pulse, in DAC full-scale units.

        ``source="memory"`` is what the FPGA would actually have played at this
        sweep point -- it is read back out of the DAC waveform memory, so it
        carries the swept scale, the detune and the phase. ``source="config"``
        recomputes from the yaml/duplicate config instead, which is the nominal
        pulse and ignores anything ``load_pulse`` overrode at runtime.

        Memory is ground truth and is never second-guessed: whatever the
        snapshot holds is what the FPGA would play, including all zeros. A pulse
        deliberately loaded at zero amplitude is a real thing a sequence does
        (tomography loads its Z point at ``scale=0``; host-side arm selection
        zeroes the pulses of the arms it is not running), and an untouched
        memory does not contain the config envelope either -- so inferring
        "never loaded" from the samples being zero, as this used to, drew
        pulses at full config amplitude that the hardware never played.

        The config is used only when the snapshot has no entry for the pulse at
        all. Whether a memory was actually written is reported separately by
        :meth:`summary` from the recorded ``load_pulse`` calls
        (:attr:`written_pulses`), where it belongs -- a scheduled pulse that was
        never loaded is a bug in the sequence, not something to paper over here.
        """
        if source == "memory":
            samples = self.loaded_envelopes.get((io_name, pulse))
            if samples is not None and len(samples):
                return samples
        return self.envelopes.get((io_name, pulse))

    @property
    def dead_time_ns(self):
        """Total inter-block dead time -- the boundary gaps from :func:`edge_gap`."""
        return sum(b.gap_after for b in self.drawn_blocks) * self.ns_per_cycle

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
                if c.symbolic and c.static_length is None:
                    shown[self.register_label(c.symbolic)] = (
                        f"{c.length * self.ns_per_cycle:.0f} ns "
                        f"({how[c.resolution]})")
            lines.append(f"  {self.unresolved} register/DSP-driven length(s): "
                         + ", ".join(f"{n} -> {v}" for n, v in sorted(shown.items())))
        # a scheduled pulse whose memory load_pulse never wrote plays whatever was
        # left in that memory, not the pulse in the yaml (rule: fpga_pulse_duplication)
        if self.written_pulses:
            unloaded = sorted({(c.io_name, c.pulse) for c in self.commands
                               if c.pulse and c.io_name and not c.is_padding
                               and (c.io_name, c.pulse) not in self.written_pulses})
            if unloaded:
                lines.append("  SCHEDULED BUT NEVER LOADED: "
                             + ", ".join(f"{io}.{pl}" for io, pl in unloaded)
                             + " -- these play whatever the memory held")
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
                           f"before the next block ({g.get('edge', '')}, "
                           f"issue {g.get('issue')} cycles)")
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
    :param register_names: ``{"REG0": "t_echo"}`` display aliases, overriding the name
        already auto-derived from the runtime's own source (the variable a
        ``Register()``/``DSP()`` was assigned to, e.g. ``rounds_reg = a.sequencer()
        .Register()`` names it ``rounds_reg`` with no need to pass this at all --
        see ``_variable_name_spy``). Only needed to relabel one, or when the source
        line isn't a plain assignment and nothing was auto-derived.
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
            "register_exprs": dict(register_exprs),
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
    written_ids = set()   # id() of every waveform memory load_pulse writes

    # Patch the class the runtime's OWN io objects use, not the installed one: with
    # use_saved_qmsmt the runtime comes from the acadia_qmsmt saved in the data folder, so
    # `acadia_qmsmt.qmsmt.InputOutput` here is a different class object and patching it would
    # silently record nothing. Walk each io's MRO to the class that actually defines
    # load_pulse, so subclasses are covered too.
    def _load_pulse_owners(rt):
        owners = {}
        ios = list(getattr(rt, "_ios", {}).values())
        for io in ios:
            for base in type(io).__mro__:
                if "load_pulse" in base.__dict__:
                    owners.setdefault(base, base.__dict__["load_pulse"])
                    break
        if not owners:      # _ios not built yet -- fall back to the installed class
            try:
                from acadia_qmsmt.qmsmt import InputOutput
                owners[InputOutput] = InputOutput.__dict__["load_pulse"]
            except Exception:
                pass
        return owners

    def _target_memory(io, memory_arg):
        """The waveform memory a load_pulse call wrote to, if resolvable.

        ``memory`` may be a pulse name, a pulse-config dict, or a
        ``WaveformMemory`` already.
        """
        if memory_arg is None:
            return None
        if getattr(memory_arg, "__array_interface__", None) is not None \
                or hasattr(memory_arg, "array"):
            return memory_arg
        name = memory_arg.get("name") if isinstance(memory_arg, dict) else memory_arg
        if not isinstance(name, str):
            return None
        return (io._pulse_cache.get(name) or {}).get("memory")

    def _make_spy(orig):
        def spy_load_pulse(self, memory=None, *a, **k):
            out = orig(self, memory, *a, **k)
            try:
                target = _target_memory(self, memory)
                if target is not None:
                    written_ids.add(id(target))
            except Exception:      # never let bookkeeping break a dry run
                pass
            return out
        return spy_load_pulse

    def on_run(self, *a, **k):
        if not capture_points:
            if state["runs"] >= point:
                raise StopDryRun()
            state["runs"] += 1
            return
        snapshots.append(_snapshot(runtime, pool, written_ids) if envelopes
                         else {"memories": {}, "written": set(),
                               "cache": _cache_words(runtime)})
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
        load_pulse_owners = _load_pulse_owners(runtime)
        for _cls, _orig in load_pulse_owners.items():
            _cls.load_pulse = _make_spy(_orig)
        source_names = {}
        # live, mutated as the sequence compiles; branch_recorder snapshots it per
        # control-flow context (see _register_spy)
        register_exprs = {}
        orig_managed_call, managed_spy, restore_loads = _register_spy(
            source_names, register_exprs, _runtime_source_file(runtime))
        ManagedResource.__call__ = managed_spy
        try:
            with hardware_stubbed(on_run, runtime=runtime,
                                  allow_instruments=allow_instruments), \
                 branch_recorder(branch_stack, register_exprs):
                try:
                    runtime.main()
                except StopDryRun:
                    pass
        finally:
            DMASynchronizer.create_schedules = staticmethod(orig_create)
            DMASynchronizer.merge_schedules = staticmethod(orig_merge)
            DMASynchronizer.__exit__ = orig_exit
            for _cls, _orig in load_pulse_owners.items():
                _cls.load_pulse = _orig
            ManagedResource.__call__ = orig_managed_call
            restore_loads()
            if restore_iterations is not _MISSING:
                runtime.iterations = restore_iterations

        if not raw_blocks:
            raise RuntimeError(
                "no synchronizer blocks were traced -- main() never reached a "
                "channel_synchronizer, or it stopped earlier than expected")

        trace = _build_trace(runtime, raw_blocks, resolve_indeterminate)
        trace.register_overrides = dict(resolve_registers or {})
        # source_names are read straight off the runtime's own Register()/DSP() call
        # sites (see _variable_name_spy); an explicit register_names entry overrides one.
        trace.register_names = {**source_names, **dict(register_names or {})}
        trace.iterations_forced = restore_iterations is not _MISSING
        trace.truncated_points = bool(max_points) and len(snapshots) >= max_points
        if capture_points:
            trace.snapshots = snapshots
            selected = point
        else:
            # only the requested point was reached, so it is the only snapshot
            trace.snapshots = [_snapshot(runtime, pool, written_ids)] if envelopes else []
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


def _snapshot(runtime, pool, written_ids=()):
    """DAC memory contents and cache words as of right now.

    ``written_ids`` are ``id()``s of the waveform memories ``load_pulse`` has
    written, collected by the spy installed in :func:`trace_runtime`. They are
    resolved to ``(io_name, pulse)`` keys here so a deliberately zero-amplitude
    pulse can be told apart from a memory nothing ever touched.
    """
    memories = {}
    written = set()
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
            if id(memory) in written_ids:
                written.add((io_name, pulse))
    return {"memories": memories, "written": written,
            "cache": _cache_words(runtime)}


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


def _is_branch(r):
    """An absolute PC branch (loop back-edge or `test` skip); target is its slot-1 immediate."""
    return r.d1 == "PC" and r.d1_minor == Destination.PC_ABSOLUTE_BRANCH


def _bus_addr(r):
    """The cache/bus address an instruction writes to BUS_ADDR (from an immediate), else None."""
    if r.d1 == "BUS_ADDR" and r.s1 == "IMM":
        return _resolve_imm(r.imm1)
    if r.d2 == "BUS_ADDR" and r.s2 == "IMM":
        return _resolve_imm(r.imm2)
    return None


def _bus_addr_pointer(r):
    """True when this instruction drives BUS_ADDR from a DSP pointer (``bus_read(pointer)``)."""
    return any(dest == "BUS_ADDR" and src == "DSP_P" for dest, src in ((r.d1, r.s1), (r.d2, r.s2)))


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
        trace.dma_timing = gateware_timing(acadia)
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
    trace.channel_nums = {str(io.channel): io.channel.num() for io in runtime._ios.values()}
    try:
        trace.cache_base = acadia._firmware.sequencer_bus_decoder["cache"].address().value()
    except Exception:
        pass
    try:
        from .machine import drain_block_issue
        trace.drain_blocks = drain_block_issue(acadia)
    except Exception:
        pass                    # the machine layout is opt-in; never fail the trace
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
                      conditional=raw.get("conditional", ()),
                      register_exprs=raw.get("register_exprs", {}))
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
                    if isinstance(raw_len, (int, np.integer)):
                        length = int(raw_len) + (1 if c.get("length_is_minus_one")
                                                 else 0)
                    else:
                        # A readable label either way (acadia's own barrier-padding
                        # arithmetic -- a channel dwelling `max(other lengths) - its
                        # own`, say -- builds an Operation tree even when every operand
                        # is already a known literal) instead of raw repr(). When it
                        # DOES fold to a plain number, that is kept alongside for
                        # register_summary() to declutter with -- see
                        # Command.static_length for why it stops there.
                        symbolic = describe_operation(raw_len)
                        folded = _operation_value(raw_len)
                        if (isinstance(folded, (int, np.integer, float, np.floating))
                                and not isinstance(folded, bool)):
                            static_length = int(folded)
                    io_name = pulse = None
                    address = c.get("address")
                    if address is not None:
                        io_name, pulse = addr_names.get(
                            (ch.num(), address), (None, None))
                    group.append(Command(
                        channel=ch_name,
                        kind=KIND.get(c["command_type"], str(c["command_type"])),
                        start=0, length=length, symbolic=symbolic,
                        static_length=static_length,
                        direct="length" not in c and "command" in c,
                        length_offset=1 if symbolic and c.get("length_is_minus_one") else 0,
                        is_padding=id(c) not in raw["user_ids"],
                        pulse=pulse, io_name=io_name, address=address))
            block.subschedules.append(group)
            block.commands.extend(group)
        trace.blocks.append(block)

    _compute_config_envelopes(runtime, trace)
    return trace


#: `gateware_timing` on the standard firmware configuration, used when it cannot be read.
DEFAULT_DMA_TIMING = {"trigger": 4, "running_seen": 1, "fifo_seen": 1, "fifo_ready": 5}


def gateware_timing(acadia):
    """Cycle offsets of the DMA trigger and status paths for this firmware configuration.

    Times are on the DMA output timeline (the DAC's own output latency is the same for
    every channel and left out):

    * ``trigger``: from the ``Trigger DMAs`` instruction to the first sample. The sequencer
      registers the bus write, the trigger dataport registers it again, then `DMA_START`.
    * ``running_seen``: from the end of a playout to the first cycle a poll reads
      ``dma_running`` low. ``running`` drops one cycle before the last sample leaves the
      output register, then passes the running dataport and the bus decoder's read register.
    * ``fifo_seen``: from the start of the descriptor whose pull changed ``fifo_empty`` /
      ``fifo_almost_empty`` to the first cycle a poll reads the new value. The pull is
      `DMA_START` cycles before that start; the FIFO flag, the DMA status register and the
      decoder each register it once.
    * ``fifo_ready``: from a FIFO push instruction to the first cycle the DMA can pull it.
    """
    firmware = acadia._firmware
    bus = firmware["sequencer_bus"]
    decoder = 1 if bus["decoder_pipeline_miso"] else 0
    trigger = bus["dma_trigger_dataport"]
    fifo_latency = max(max(firmware["rfdc"][kind]["dma_fifo_latency"]) for kind in ("dac", "adc"))
    return {
        "trigger": (1 + max(trigger["pipeline"]) + (1 if trigger["bus_pipeline"] else 0)
                    + DMA_START),
        "running_seen": max(bus["dma_running_dataport"]["pipeline"]) + decoder - 1,
        "fifo_seen": 2 + decoder - DMA_START,
        "fifo_ready": 1 + (1 if any(bus["dma_pipeline"]) else 0) + fifo_latency,
    }


def sequencer_control_flow(acadia):
    """Trigger/poll addresses and branches, for costing an executed edge.

    Straight-line edges can be costed by counting instructions between the poll and the
    next trigger. A loop back-edge cannot: execution leaves the poll, reaches the branch,
    jumps to the branch target, and runs forward from there to that block's trigger. So the
    branch and its target address are needed as well.

    The first DMA FIFO push per channel ahead of each trigger is kept too: a descriptor that
    lands while the channel is still playing continues without the trigger.

    :return: ``{"triggers": [addr], "polls": {nth trigger: addr}, "branches": [(addr, target)],
                "back_branches": [(branch addr, target addr)],
                "pushes": {nth trigger: {channel: addr}}}``
    """
    prog = decode_program(acadia)

    triggers = [r.i for r in prog if _is_trigger(r)]
    polls = {}
    for nth, trigger in enumerate(triggers):
        limit = triggers[nth + 1] if nth + 1 < len(triggers) else len(prog)
        poll = next((j for j in range(trigger + 1, limit)
                     if _is_dma_poll(prog[j])), None)
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

    # a FIFO push writes one of the four command words below a DMA's base; base + 4 resets it
    decoder = acadia._firmware.sequencer_bus_decoder
    dma_base = {}
    for name in decoder.keys():
        match = re.fullmatch(r"(dac|adc)(\d+)_dma", name)
        if match:
            dma_base[decoder[name].address().value()] = f"{match.group(1).upper()}{match.group(2)}"
    pushes = {}
    for nth, trigger in enumerate(triggers):
        first = {}
        for j in range(triggers[nth - 1] + 1 if nth else 0, trigger):
            address = _bus_addr(prog[j])
            if address is not None and prog[j].d2 == "BUS_DATA" and not address & 4:
                channel = dma_base.get(address & ~3)
                if channel is not None:
                    first.setdefault(channel, j)
        pushes[nth] = first

    return {"triggers": triggers, "polls": polls, "branches": branches,
            "back_branches": [(b, t) for b, t in branches if t < b], "pushes": pushes}


def edge_gap(control_flow, from_nth, to_nth, start=None):
    """Dead cycles between two executed blocks, and how it breaks down.

    ``from_nth``/``to_nth`` index the compiled program's trigger list. A backward edge picks
    up the branch instructions and the taken-branch penalty. ``start`` walks from that address
    instead of the block's blocking poll (a drain's own poll, or past a batch's trigger).
    """
    triggers, polls = control_flow["triggers"], control_flow["polls"]
    if start is None and from_nth not in polls:
        return None                                  # non-blocking: no wait, no gap
    if not (0 <= to_nth < len(triggers)):
        return None
    poll = polls[from_nth] if start is None else start

    if to_nth == from_nth + 1:
        issue = triggers[to_nth] - poll + 1
        penalty = 0
        kind = "fall-through"
    else:
        # The executed path leaves address order: walk it instruction by instruction from the
        # poll to the target block's trigger, deciding each branch by where that block is. A
        # forward branch (a `test` guard) is taken unless the target lies inside what it skips;
        # a backward branch (a loop's back-edge) is taken when the target lies inside the loop.
        # Chains of skipped tests, a loop exit, and a skip followed by a back-edge all follow.
        goal, branch_at = triggers[to_nth], dict(control_flow["branches"])
        pc, issue, kinds = poll, 0, []
        for _ in range(4 * (max(triggers) + len(branch_at) + 2)):
            if pc == goal:
                break
            issue += 1
            target = branch_at.get(pc)
            if target is not None and ((target > pc and not pc < goal < target)
                                       or (target <= pc and target <= goal <= pc)):
                kinds.append("loop back" if target <= pc else "skip")
                pc = target
            else:
                pc += 1
        else:
            return None
        issue += 1                                   # the target trigger is fetched too
        penalty = TAKEN_BRANCH_CYCLES * len(kinds)
        kind = (f"taken branch ({kinds[0]})" if len(kinds) == 1
                else f"taken branches ({', '.join(kinds)})" if kinds else "fall-through")
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
        address = pointer = None
        for j in range(r.i - 1, max(r.i - 12, -1), -1):
            pointer = _bus_addr_pointer(prog[j])
            if pointer:
                break
            found = _bus_addr(prog[j])
            if found is not None:
                address = found
                break
        # a DSP unit's output DSPn drives the command length, not DSP_ABn (its input)
        major, minor = loaded
        name = f"DSP{minor}" if major == "DSP_AB" else f"REG{minor}"
        if pointer:
            # bus_read(pointer): the address is a DSP walking the cache, not a literal
            # (an earlier literal BUS_ADDR would name an unrelated device)
            registers[name] = {"source": "cache[pointer]", "cache_word": None}
            continue
        if address is None:
            continue
        if cache_base <= address < cache_base + cache_words:
            word = address - cache_base
            registers[name] = {"source": f"cache[{word}]", "cache_word": word}
        else:
            registers[name] = {
                "source": devices.get(address, f"bus 0x{address:X}"),
                "cache_word": None}
    return registers


def _dsp_pointer_config(r):
    """The DSP index a ``P+1`` config is written to (the walking pointer), else None.
    ``P+1`` means "from now on only increment", so this marks the cache-stream pointer DSP."""
    for dest, minor, imm in ((r.d1, r.d1_minor, r.imm1), (r.d2, r.d2_minor, r.imm2)):
        if dest == "DSP_CFG":
            mode = getattr(imm, "mode", None)
            if mode == "P+1" or getattr(mode, "name", None) == "P+1":
                return minor
    return None


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

    # the walking pointer: the DSP reconfigured P+1 (loaded once, then only incremented)
    cfg = next((c for r in prog if (c := _dsp_pointer_config(r)) is not None), None)
    if cfg is None:
        return None

    decoder = acadia._firmware.sequencer_bus_decoder
    cache_base = decoder["cache"].address().value()
    cache_words = acadia._firmware["sequencer_cache_memory"]["size_bits"] // 8

    # start_offset: the immediate loaded into that DSP's AB input (its first cache address)
    start_offset = None
    for r in prog:
        addr = None
        for dest, minor, src, imm in ((r.d1, r.d1_minor, r.s1, r.imm1),
                                      (r.d2, r.d2_minor, r.s2, r.imm2)):
            if dest == "DSP_AB" and minor == cfg and src == "IMM":
                addr = _resolve_imm(imm)
        if addr is not None:
            if cache_base <= addr < cache_base + cache_words:
                start_offset = addr - cache_base
            break

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
    # cost of its back-edge (the fifo-refill floor). Use the back-edge whose span contains the
    # direct command.
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
            floor = (branch - target + 1) + TAKEN_BRANCH_CYCLES
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
