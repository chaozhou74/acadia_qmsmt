"""
Rebuild a runtime from a saved data folder.

A folder produced by ``Runtime.deploy`` is self-describing:

===================== ====================================================
``kwargs.json``       every field; IOConfig fields are stored as *resolved*
                      config dicts (see ``QMsmtRuntime._dump_fields``), so
                      no yaml lookup and no path fixups are needed
``runtime.py``        the runtime source exactly as it was that day
``acadia_qmsmt.py``   the qmsmt library exactly as it was that day
``compiled.log``      the sequencer program that actually ran
===================== ====================================================

Loading is delegated to ``acadia_qmsmt.utils.saved_runtime_loader``, which is the
same path acadia_gui's ``LivePlotWidget`` uses -- so a folder that opens in the
GUI traces here, and vice versa.
"""
import logging
from pathlib import Path

logger = logging.getLogger("sequence_viz")


def is_data_folder(folder):
    """True if ``folder`` looks like a deployed data folder."""
    try:
        folder = Path(folder)
    except TypeError:
        return False
    return (folder / "kwargs.json").is_file() and (folder / "runtime.py").is_file()


def load_runtime(folder, use_saved_qmsmt=True, overrides=None):
    """Reconstruct the runtime object a data folder was produced by.

    :param use_saved_qmsmt: import against the ``acadia_qmsmt.py`` saved in the
        folder rather than the installed package — the faithful choice for an
        archived run. If that import fails the installed package is used instead,
        matching how acadia_gui's ``LivePlotWidget`` handles it.
    :param overrides: field values to replace after construction, e.g. shrinking
        a sweep so the dry run is quick.
    :return: ``(runtime, class_name, used_saved_qmsmt)``
    """
    from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir

    folder = Path(folder)
    if not is_data_folder(folder):
        raise FileNotFoundError(
            f"{folder} is not a deployed data folder "
            f"(needs kwargs.json and runtime.py)")

    used_saved = bool(use_saved_qmsmt)
    if used_saved:
        try:
            runtime = load_runtime_from_data_dir(str(folder), use_saved_qmsmt=True)
        except Exception as exc:
            logger.warning("Failed to load the acadia_qmsmt.py saved in %s (%s); "
                           "falling back to the installed package", folder, exc,
                           exc_info=True)
            used_saved = False
    if not used_saved:
        runtime = load_runtime_from_data_dir(str(folder), use_saved_qmsmt=False)

    for name, value in (overrides or {}).items():
        setattr(runtime, name, value)

    return runtime, type(runtime).__name__, used_saved


def run_failed_the_same_way(folder, exc):
    """The run's OWN board-side error, when the folder failed for the same reason we just did.

    A trace re-executes ``runtime.main()``, so a sequence that could not compile ON THE BOARD
    cannot compile here either -- and the viewer then shows a raw exception that reads like a
    viewer bug. It usually is not: the folder is the wreckage of a run that never produced data.
    Returned so the caller can say which it is instead of leaving the user to guess.

    Matches on the exception's own text appearing in the run's board-side log, which is specific
    enough not to fire on an unrelated failure.
    """
    from pathlib import Path

    import re

    # Object reprs carry a memory ADDRESS, which differs between the board's run and ours
    # ("<WaveformMemory object at 0xffff75029570>" vs "... at 0x7cedff3132e0>"). Comparing the
    # raw text therefore never matches on exactly the errors worth reporting, so normalise the
    # addresses away on both sides.
    def normalise(text):
        return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)

    message = normalise(str(exc)).split("\n")[0].strip()
    if len(message) < 6:
        return None
    for name in ("remote_main.log", "remote_stderr.log", "runtime.log"):
        log = Path(folder) / name
        try:
            text = normalise(log.read_text(errors="ignore"))
        except OSError:
            continue
        # The message must appear INSIDE a traceback in the run's own log, not merely somewhere
        # in it. That is what makes a short message safe to attribute: "Empty synchronizer" is
        # only 18 characters, but the same exception raised at the same point in the same
        # sequence is not a coincidence.
        #
        # A blunt length threshold was tried first and was wrong in exactly the interesting
        # direction: it rejected every "Empty synchronizer" -- 11 of 400 archived runs -- so the
        # viewer reproduced the board's own failure faithfully and then reported it as if it
        # were the viewer's fault.
        head, _, tail = text.partition("Traceback")
        if tail and message in tail:
            return name
    return None


def trace_folder(folder, point=0, resolve_indeterminate=0, envelopes=True,
                 use_saved_qmsmt=True, overrides=None, **trace_kwargs):
    """Load a data folder and trace the sequence it ran. Returns a SequenceTrace.

    If the sequence cannot be compiled, the error is re-raised with a note saying whether the
    ORIGINAL run hit the same error on the board -- i.e. whether this is a viewer problem or a
    faithful replay of a run that already failed.
    """
    from .tracing import trace_runtime

    runtime, class_name, used_saved = load_runtime(
        folder, use_saved_qmsmt=use_saved_qmsmt, overrides=overrides)
    try:
        trace = trace_runtime(runtime, point=point,
                              resolve_indeterminate=resolve_indeterminate,
                              envelopes=envelopes, **trace_kwargs)
    except Exception as exc:
        where = run_failed_the_same_way(folder, exc)
        if where:
            raise type(exc)(
                f"{exc}\n\n[sequence_viz] This is NOT a viewer error: the original run failed "
                f"the same way on the board -- the identical message is in {where}, and the "
                f"folder holds no data. Fix the sequence/config and re-run; the viewer is "
                f"replaying a run that never compiled."
            ) from exc
        raise
    trace.runtime_class = class_name
    trace.source = str(folder)
    trace.used_saved_qmsmt = used_saved
    return trace
