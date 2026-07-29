"""
sequence_viz -- visualize the pulse sequence a QMsmtRuntime compiles.

Runs the runtime off-hardware up to the first ``acadia.run()``, hooks acadia's
own barrier-alignment routine, and renders the resulting per-channel schedule.

Two entry points::

    from sequence_viz import plot_folder, plot_runtime

    fig, ax, trace = plot_folder("/path/to/data_folder")   # archived run
    fig, ax, trace = plot_runtime(rt)                      # live object

Self-contained: depends only on ``acadia``, ``acadia_qmsmt``, ``numpy`` and
``matplotlib`` (plus PyQt5 in ``qt_widget`` alone), so the whole directory can
live inside ``acadia_qmsmt`` unchanged.
"""
from .compiled_log import compare as compare_with_compiled_log
from .compiled_log import parse as parse_compiled_log
from .dryrun import (InstrumentAccessBlocked, StopDryRun, fake_attach,
                     hardware_stubbed)
from .folder import is_data_folder, load_runtime, trace_folder
from .interactive import SequenceView, interactive_view
from .plotting import (assign_colors, base_pulse, branch_caption,
                       branch_regions, draw, plot_trace)
from .tracing import (Block, Command, SequenceTrace, describe_registers,
                      trace_runtime)

__all__ = [
    "plot_folder", "plot_runtime", "explore_folder", "explore_runtime",
    "trace_folder", "trace_runtime", "plot_trace", "draw",
    "SequenceView", "interactive_view",
    "assign_colors", "base_pulse", "branch_regions", "branch_caption",
    "load_runtime", "is_data_folder",
    "parse_compiled_log", "compare_with_compiled_log",
    "SequenceTrace", "Block", "Command", "describe_registers",
    "fake_attach", "hardware_stubbed", "StopDryRun",
    "InstrumentAccessBlocked",
]


def plot_folder(folder, point=0, resolve_indeterminate=0, use_saved_qmsmt=True,
                overrides=None, **plot_kwargs):
    """Trace an archived data folder and plot it. Returns ``(fig, ax, trace)``."""
    trace = trace_folder(folder, point=point,
                         resolve_indeterminate=resolve_indeterminate,
                         use_saved_qmsmt=use_saved_qmsmt,
                         overrides=overrides)
    fig, ax = plot_trace(trace, **plot_kwargs)
    return fig, ax, trace


def plot_runtime(runtime, point=0, resolve_indeterminate=0, **plot_kwargs):
    """Trace a live runtime object and plot it. Returns ``(fig, ax, trace)``."""
    trace = trace_runtime(runtime, point=point,
                          resolve_indeterminate=resolve_indeterminate)
    trace.source = "live runtime"
    fig, ax = plot_trace(trace, **plot_kwargs)
    return fig, ax, trace


def explore_folder(folder, point=0, resolve_indeterminate=0,
                   use_saved_qmsmt=True, overrides=None, trace_kwargs=None,
                   **view_kwargs):
    """Trace a data folder and open it in an interactive view (drag to zoom).

    Notebooks need a widget backend first: ``%matplotlib widget``.
    Returns the :class:`SequenceView`; its ``.trace`` holds the data.
    """
    trace = trace_folder(folder, point=point,
                         resolve_indeterminate=resolve_indeterminate,
                         use_saved_qmsmt=use_saved_qmsmt,
                         overrides=overrides, **(trace_kwargs or {}))
    return interactive_view(trace, **view_kwargs)


def explore_runtime(runtime, point=0, resolve_indeterminate=0, **view_kwargs):
    """Trace a live runtime and open it in an interactive view (drag to zoom)."""
    trace = trace_runtime(runtime, point=point,
                          resolve_indeterminate=resolve_indeterminate)
    trace.source = "live runtime"
    return interactive_view(trace, **view_kwargs)
