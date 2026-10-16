"""
Run a QMsmtRuntime off-hardware.

``Acadia.__init__`` touches no hardware -- every ``/dev/mem`` mapping happens in
``attach()``. So a runtime can be constructed, compiled and inspected on any
machine as long as the hardware entry points are neutralised first.

``fake_attach`` backs every managed memory pool with a host ``bytearray`` instead
of an mmap, which keeps ``load_pulse``, ``load_windows`` and cache writes working
-- they simply write into host memory. Everything that talks to the RFDC, the
NCOs or the sequencer GPIO is replaced with a no-op.
"""
import logging
import tempfile
from contextlib import contextmanager
from itertools import count

import numpy as np

from acadia.system import Acadia

logger = logging.getLogger(__name__)


class StopDryRun(Exception):
    """Raised in place of ``Acadia.run()`` to end a dry run at the first point."""


class InstrumentAccessBlocked(Exception):
    """The runtime tried to reach the instrument server during a dry run.

    A dry run must have no effect on the lab. Some runtimes set *external*
    instruments in ``main()`` -- a flux-sweep spectroscopy ramps a current source
    through ``instrumentserver``'s proxy client -- and those calls are real: they
    move real hardware, on a fridge that may be mid-experiment. So the client is
    blocked rather than allowed through, and tracing such a runtime fails loudly
    instead of quietly touching the instruments.

    Pass ``allow_instruments=True`` to ``trace_runtime``/``trace_folder`` only if
    you are certain the instrument server is offline or the writes are harmless.
    """


def fake_attach(self):
    """Stand-in for ``Acadia.attach`` that maps host buffers instead of /dev/mem.

    Sized to what the sequence actually allocated rather than to the pool's
    address-space limit -- the two PL-DDR pools and PS-DDR alone declare 9.6 GB,
    and zeroing that costs seconds per trace. attach() only assigns
    ``__array_interface__`` at each instance's offset, so a buffer covering the
    highest allocated instance is sufficient; like the real ``_attach_resource``,
    memories allocated after attach are not mapped either way.
    """
    self._mem_maps = []
    self._fake_buffers = []

    def attach_pool(pool):
        needed = max((inst._resource_id + inst._resource_size
                      for inst in pool.instances), default=0)
        if not needed:
            return                      # nothing allocated in this pool
        buf = bytearray(needed + (1 << 16))     # slack for late allocations
        self._fake_buffers.append(buf)
        pool.attach(buf)
        for inst in pool.instances:
            ai = inst.__array_interface__
            inst._array = np.frombuffer(
                ai["data"], dtype=np.dtype(ai["typestr"]),
                count=inst.size, offset=ai["offset"]).reshape(inst.shape)

    pools = [self.CacheArray, self.OCMArray,
             self.PLDDR0Array, self.PLDDR1Array, self.PSDDRArray]
    pools += list(self.DACArray) + list(self.CMACCKernelArray)
    for pool in pools:
        attach_pool(pool)

    self._sequencer_instruction_memory = bytearray(
        self._firmware["sequencer_instruction_memory"]["size_bits"] // 8)


def _defining_class(cls, attribute):
    """The class in ``cls``'s MRO that actually defines ``attribute``.

    Needed because a runtime loaded with the archived ``acadia_qmsmt.py``
    subclasses *that* module's ``QMsmtRuntime``, which is a different class
    object from the installed one -- patching only the installed class would
    leave the archived hardware calls live.
    """
    if cls is None:
        return None
    for klass in cls.__mro__:
        if attribute in klass.__dict__:
            return klass
    return None


def _frequency_setters(*anchors):
    """Every class defining ``set_frequency`` in the modules ``anchors`` come from.

    ``set_frequency`` reaches straight through to ``Channel.set_nco_frequency``, which
    raises on a host machine, so all of them have to be neutralised. ``InputOutput`` is
    not the only owner: ``MeasurableResonator`` and ``Qubit`` define their own, and a
    spectroscopy runtime sweeps the NCO through one of *those*. Discovered by module
    rather than by a fixed list so the archived ``acadia_qmsmt.py``'s own copies are
    found too -- patching the installed classes does nothing for a runtime loaded
    against the archive -- and so a class added later needs no change here.
    """
    import inspect
    import sys

    found, seen = [], set()
    for anchor in anchors:
        for klass in getattr(anchor, "__mro__", [anchor]):
            module = sys.modules.get(getattr(klass, "__module__", None))
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            found += [obj for obj in vars(module).values()
                      if inspect.isclass(obj) and "set_frequency" in obj.__dict__]
    return found


OPERATOR_SYMBOLS = {
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "and_": "&", "or_": "|", "xor": "^", "add": "+", "sub": "-", "invert": "~",
}


def _describe_condition(value, limit=56):
    """Render a sequencer condition readably.

    ``Operation.__str__`` gives ``Operation(<built-in function eq>, (REG0, 0), {})``;
    this turns nested operations back into ``REG0 == 0``.
    """
    from acadia.compiler import Operation

    def render(node):
        if not isinstance(node, Operation):
            return str(node)
        symbol = OPERATOR_SYMBOLS.get(getattr(node._op, "__name__", ""), None)
        args = [render(a) for a in node._args]
        if symbol and len(args) == 2:
            return f"{args[0]} {symbol} {args[1]}"
        if symbol and len(args) == 1:
            return f"{symbol}{args[0]}"
        name = getattr(node._op, "__name__", str(node._op))
        return f"{name}({', '.join(args)})"

    text = " ".join(render(value).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


@contextmanager
def branch_recorder(stack):
    """Track which control-flow blocks the sequence is inside while it compiles.

    ``test``, ``repeat_until`` and ``loop`` are ``@contextmanager`` generators, so
    their bodies run **once**, in Python, at compile time -- a conditional pulse
    records into the synchronizer exactly like an unconditional one. Wrapping them
    lets the trace at least know which commands are conditional, and on what.

    ``speculation`` is carried through because it decides the code layout: with
    ``True`` the body is inline and the branch jumps past it when the condition
    fails; with ``False`` the body is out-of-line and the taken path costs two
    extra instructions. Execution order is then not address order.
    """
    from acadia.sequencer import Sequencer

    originals = {}
    entry_counter = count()

    def wrap(name, describe):
        original = getattr(Sequencer, name)
        originals[name] = original

        @contextmanager
        def wrapped(self, *args, **kwargs):
            context = describe(args, kwargs)
            # One id per `with` entry, so two control-flow blocks that describe IDENTICALLY are
            # still distinguishable. Sibling loops with the same condition are routine: cooling
            # two qubits emits two `repeat_until(DSP0 == 1)` blocks in a row, and without an id
            # SequenceTrace.execution_plan would merge them into one body and replay the pair
            # together instead of each on its own count. The bodies run once here, at compile
            # time, so one id per entry is exactly one id per control-flow block.
            context["id"] = next(entry_counter)
            stack.append(context)
            try:
                with original(self, *args, **kwargs) as value:
                    yield value
            finally:
                stack.pop()

        setattr(Sequencer, name, wrapped)

    wrap("test", lambda a, k: {
        "kind": "test",
        "condition": _describe_condition(a[0] if a else k.get("condition")),
        "speculation": k.get("speculation", a[2] if len(a) > 2 else True)})
    wrap("repeat_until", lambda a, k: {
        "kind": "repeat_until",
        "condition": _describe_condition(a[0] if a else k.get("condition")),
        "speculation": None})
    def loop_count(args):
        """Iterations, using the same signatures as ``range`` (see Sequencer.loop)."""
        try:
            if not args:
                return None                      # loop() runs forever
            return len(range(*[int(x) for x in args]))
        except (TypeError, ValueError):
            return None

    wrap("loop", lambda a, k: {
        "kind": "loop",
        "condition": f"range{a}" if a else "forever",
        "count": loop_count(a),
        "speculation": None})
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(Sequencer, name, original)


#: Set by :func:`numeric_check_shim` when it had to stand in for a stale ``acadia.is_numeric``.
#: Read by the tracer so ``trace.acadia_shims`` / ``summary()`` can say the station needs updating.
NUMERIC_CHECK_SHIM_NOTE = (
    "installed acadia predates the fix for numeric checking of Operations with arbitrary "
    "functions (acadia commit 370fc87, 2026-07-08). Without it, is_numeric() compares "
    "Operation._op (a FUNCTION) against Operable.NUMERIC_OPERATORS (a dict keyed by STRINGS), so "
    "NO Operation is ever numeric -- not even add/sub. A barrier that aligns 2+ occupied channels "
    "emits Operation(max, len_a, len_b) as its alignment dwell; classified non-numeric it is "
    "pushed onto a DSP, where max has no lowering, and compile dies with 'Unable to find a DSP "
    "configuration for Operation(<built-in function max>, ...)'. The BOARD, running current "
    "acadia, resolves that dwell as an assembly-time immediate and runs the sequence fine. "
    "sequence_viz stood in for the fixed function FOR THIS TRACE ONLY, so the picture matches the "
    "board -- but every other client-side acadia.compile() on this station still fails. "
    "Fix the station: reinstall acadia from its source repo (e.g. "
    "`$ACADIA_ENV/bin/pip install -e /path/to/acadia/pyacadia`), or copy the current "
    "acadia/sequencer.py over the installed one -- it is the only pure-Python file that differs.")


def _is_numeric_fixed(obj):
    """``acadia.sequencer.is_numeric`` as of acadia commit 370fc87, reproduced verbatim in effect.

    An ``Operation`` is numeric when all of its arguments are; the *function* is not required to be
    in a whitelist (the point of that commit -- ``max``/``min`` and any other numeric callable are
    legitimate). Deliberately a faithful stand-in rather than an improvement: this must not make
    the trace disagree with a correctly-installed acadia.
    """
    from acadia.compiler import Operation
    from acadia.sequencer import DSPConfiguration, ProcessorInstruction, Symbol

    for t in (int, bool, DSPConfiguration, ProcessorInstruction):
        if isinstance(obj, t) or (isinstance(obj, Symbol)
                                  and (obj.value_type() is t
                                       or t in obj.value_type().__bases__)):
            return True
    if isinstance(obj, Operation):
        return (all(_is_numeric_fixed(a) for a in obj._args)
                and all(_is_numeric_fixed(v) for v in obj._kwargs.values()))
    try:
        int(obj)
        return True
    except (TypeError, ValueError):
        return False


def numeric_check_shim(patch):
    """Stand in for a STALE ``acadia.is_numeric`` for the duration of a dry run.

    Returns the note describing what was shimmed, or ``None`` when the installed acadia is already
    correct -- in which case nothing at all is patched.

    Detected behaviourally, not by version: a correct ``is_numeric`` says ``Operation(sub, 1, 1)``
    is numeric and the stale one does not. So this self-disables the moment the station is updated,
    and it cannot mask a *future* acadia whose semantics differ for some other reason.

    Patched in BOTH ``acadia.sequencer`` and ``acadia.system``: the internal call sites resolve the
    module global at call time, but ``system.py`` does ``from .sequencer import is_numeric``, which
    binds the function object at import -- so patching only the sequencer module would leave
    ``system``'s copy stale.
    """
    from acadia import sequencer, system
    from acadia.compiler import Operable, Operation

    try:
        probe = Operation(Operable.NUMERIC_OPERATORS["sub"], 1, 1)
        if sequencer.is_numeric(probe):
            return None                       # installed acadia is current -- patch nothing
    except Exception:                         # unknown acadia shape; leave it alone
        return None

    logger.warning("sequence_viz: %s", NUMERIC_CHECK_SHIM_NOTE)
    for module in (sequencer, system):
        if getattr(module, "is_numeric", None) is not None:
            patch(module, "is_numeric", staticmethod(_is_numeric_fixed).__func__)
    return NUMERIC_CHECK_SHIM_NOTE


def already_traced(runtime):
    """True if ``main()`` has already been run on this runtime's Acadia."""
    sequencers = getattr(getattr(runtime, "acadia", None), "_sequencer_type", None)
    return bool(sequencers is not None and sequencers.instances)


@contextmanager
def preserved_runtime_state(runtime):
    """Undo the runtime-object mutations that running ``main()`` causes.

    ``main()`` is not read-only: ``duplicate_pulse`` appends a uniquely-named entry
    to ``io._config["pulses"]`` on every call, and pulse caches and ADC memories
    fill in. That matters because ``Runtime._dump_fields`` serialises the *live*
    ``io._config`` into ``kwargs.json``, so tracing a runtime and then deploying
    the same object would archive the tracer's leftover duplicates.

    Restores the pulse mapping, the pulse cache and the allocated ADC memories.
    Allocations inside the Acadia object itself (sequencer programs, waveform
    memory) cannot be undone -- which is why a traced runtime still must not be
    traced again; see :func:`already_traced`.
    """
    added = [name for name in ("data_manager", "local_directory")
             if not hasattr(runtime, name)]
    saved = []
    for io in getattr(runtime, "_ios", {}).values():
        config = getattr(io, "_config", None)
        saved.append((
            io,
            dict(config.get("pulses", {})) if isinstance(config, dict) else None,
            dict(getattr(io, "_pulse_cache", {}) or {}),
            dict(getattr(io, "_allocated_memories", {}) or {}),
        ))
    try:
        yield
    finally:
        for io, pulses, pulse_cache, memories in saved:
            if pulses is not None:
                io._config["pulses"] = pulses
            if hasattr(io, "_pulse_cache"):
                io._pulse_cache.clear()
                io._pulse_cache.update(pulse_cache)
            if hasattr(io, "_allocated_memories"):
                io._allocated_memories.clear()
                io._allocated_memories.update(memories)
        for name in added:
            if hasattr(runtime, name):
                delattr(runtime, name)


@contextmanager
def hardware_stubbed(on_run, runtime=None, allow_instruments=False):
    """Patch out every hardware entry point for the duration of the context.

    :param on_run: called instead of ``Acadia.run``; raise :class:`StopDryRun`
        from it to end the dry run.
    :param runtime: the runtime about to be traced. Its own classes are patched
        in addition to the installed ones, so runtimes loaded against an
        archived ``acadia_qmsmt`` are covered too.
    :param allow_instruments: let the runtime reach the instrument server. Off by
        default -- see :class:`InstrumentAccessBlocked`.
    """
    from acadia_qmsmt.qmsmt import InputOutput, QMsmtRuntime

    saved = {}

    def patch(obj, name, fn):
        if obj is None or not hasattr(obj, name) or (obj, name) in saved:
            return
        saved[(obj, name)] = getattr(obj, name)
        setattr(obj, name, fn)

    noop = lambda self, *a, **k: None

    # Compatibility, NOT a workaround for a defect in current acadia: if this station's acadia is
    # older than the numeric-check fix, a legal barrier that the board runs fine cannot be compiled
    # here at all. Restore the fixed behaviour for this trace so the picture matches the board, and
    # warn so the station gets updated. Silent no-op on a current acadia.
    numeric_check_shim(patch)

    runtime_classes = [QMsmtRuntime]
    acadia_classes = [Acadia]
    serve_classes = [_defining_class(QMsmtRuntime, "final_serve")]
    io_classes = list(_frequency_setters(InputOutput))
    if runtime is not None:
        runtime_classes.append(_defining_class(type(runtime), "configure_channels"))
        serve_classes.append(_defining_class(type(runtime), "final_serve"))
        acadia_classes.append(type(getattr(runtime, "acadia", None)))
        # the archived acadia_qmsmt defines its own copies of these classes
        io_classes += _frequency_setters(type(runtime))
        for io in getattr(runtime, "_ios", {}).values():
            io_classes.append(_defining_class(type(io), "set_frequency"))

    # Acadia.compile writes compiled.log into the working directory; send it to
    # a temp dir so a dry run leaves nothing behind
    scratch = tempfile.TemporaryDirectory(prefix="sequence_viz_")
    original_compile = Acadia.compile

    def compile_to_scratch(self, sequence, overwrite=False, output_directory=None):
        return original_compile(self, sequence, overwrite=overwrite,
                                output_directory=output_directory or scratch.name)

    for klass in acadia_classes:
        patch(klass, "compile", compile_to_scratch)
        patch(klass, "attach", fake_attach)
        patch(klass, "assemble", lambda self, *a, **k: {})
        patch(klass, "load", noop)
        patch(klass, "detach", noop)
        patch(klass, "align_tile_latencies", noop)
        patch(klass, "update_ncos_synchronized", noop)
        patch(klass, "reset_nco_phase", noop)
        patch(klass, "update_nco_phase", noop)
        patch(klass, "reset_logic", noop)
        patch(klass, "run", on_run)
    for klass in runtime_classes:
        patch(klass, "configure_channels", noop)
    # final_serve spins on DataManager.serve() waiting for a consumer that will
    # never connect off-hardware -- 3.7M calls and ~5 s per trace
    for klass in serve_classes:
        patch(klass, "final_serve", noop)
    # Stubbing `final_serve` is not enough: plenty of archived runtimes call `self.data.serve()`
    # STRAIGHT from main()'s sweep loop (e.g. DualRail_echo .../runtime.py:1906), and that call
    # BINDS A REAL SOCKET. A dry run is supposed to leave nothing behind, and this one did not --
    # tracing two runtimes at once failed 17 of 153 times with "ValueError: Failed to bind to
    # socket" while the same 17 passed alone, and a trace would equally fail whenever a live
    # measurement holds the port.
    #
    # DataManager is an immutable C type, so its methods cannot be patched (attempting it raises
    # "cannot set 'serve' attribute of immutable type"). Wrap the runtime's INSTANCE instead: a
    # proxy that swallows serve/sync and forwards everything else untouched.
    if runtime is not None and getattr(runtime, "data_manager", None) is not None:
        real_manager = runtime.data_manager

        class _QuietDataManager:
            """Forwards to the real DataManager but never touches the network.

            The dunders are spelled out on purpose: Python looks special methods up on the TYPE,
            not through ``__getattr__``, and runtimes index the manager directly
            (``self.data["q"].write(...)``), so a proxy without ``__getitem__`` breaks main()
            rather than quietly forwarding.
            """

            def serve(self, *a, **k):
                return None

            def sync(self, *a, **k):
                return None

            def __getattr__(self, name):
                return getattr(real_manager, name)

            def __getitem__(self, key):
                return real_manager[key]

            def __setitem__(self, key, value):
                real_manager[key] = value

            def __contains__(self, key):
                return key in real_manager

            def __iter__(self):
                return iter(real_manager)

            def __len__(self):
                return len(real_manager)

        patch(runtime, "data_manager", _QuietDataManager())
    for klass in io_classes:
        patch(klass, "set_frequency", noop)

    # InputOutput.__getattr__ forwards any unknown attribute to the underlying Channel -- a
    # C extension type whose NCO/phase/frequency methods raise "... may only be called on
    # RFSoC hardware" off the board. A runtime that calls a Channel method straight through
    # the io in main() (e.g. MeasurableResonator.reset_nco_phase -> io.reset_nco_phase())
    # bypasses the set_frequency stub above and aborts the trace. Channel is immutable so it
    # can't be patched; instead swap __getattr__ for one that forwards as usual but turns a
    # forwarded call hitting that guard into a no-op -- generically, for any such method.
    warned = set()

    def hardware_safe_getattr(self, name):
        attr = getattr(self._channel, name)      # same lookup the real __getattr__ does
        if not callable(attr):
            return attr

        def hardware_noop(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except SystemError as exc:
                if "RFSoC hardware" in str(exc):
                    if name not in warned:       # once per method, not once per call
                        warned.add(name)
                        logger.warning(
                            "dry run: skipped RFSoC-only Channel.%s() (returned None); "
                            "off-hardware trace only", name)
                    return None
                raise

        return hardware_noop

    getattr_classes = [_defining_class(InputOutput, "__getattr__")]
    if runtime is not None:
        for io in getattr(runtime, "_ios", {}).values():
            getattr_classes.append(_defining_class(type(io), "__getattr__"))
    for klass in dict.fromkeys(getattr_classes):     # de-dup, keep order
        patch(klass, "__getattr__", hardware_safe_getattr)

    if not allow_instruments:
        def blocked(self, *a, **k):
            raise InstrumentAccessBlocked(
                "this runtime opens an instrumentserver client in main(), so tracing "
                "it would send real commands to real instruments (a flux-sweep "
                "spectroscopy ramps a bias source this way). Tracing was stopped "
                "before any connection was made. If you are sure the writes are "
                "harmless -- e.g. the instrument server is not running -- re-trace "
                "with allow_instruments=True.")

        try:
            from instrumentserver.client.proxy import Client as InstrumentClient
        except Exception:
            InstrumentClient = None
        patch(InstrumentClient, "__init__", blocked)

    try:
        yield
    finally:
        for (obj, name), fn in saved.items():
            setattr(obj, name, fn)
        scratch.cleanup()
