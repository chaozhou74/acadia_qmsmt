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


def trace_folder(folder, point=0, resolve_indeterminate=0, envelopes=True,
                 use_saved_qmsmt=True, overrides=None, **trace_kwargs):
    """Load a data folder and trace the sequence it ran. Returns a SequenceTrace."""
    from .tracing import trace_runtime

    runtime, class_name, used_saved = load_runtime(
        folder, use_saved_qmsmt=use_saved_qmsmt, overrides=overrides)
    trace = trace_runtime(runtime, point=point,
                          resolve_indeterminate=resolve_indeterminate,
                          envelopes=envelopes, **trace_kwargs)
    trace.runtime_class = class_name
    trace.source = str(folder)
    trace.used_saved_qmsmt = used_saved
    return trace
