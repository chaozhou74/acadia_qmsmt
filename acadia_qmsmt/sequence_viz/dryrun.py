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
import tempfile
from contextlib import contextmanager

import numpy as np

from acadia.system import Acadia


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

    def wrap(name, describe):
        original = getattr(Sequencer, name)
        originals[name] = original

        @contextmanager
        def wrapped(self, *args, **kwargs):
            stack.append(describe(args, kwargs))
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
    for klass in io_classes:
        patch(klass, "set_frequency", noop)

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
