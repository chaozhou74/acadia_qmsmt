"""
A drop-in PyQt5 widget for acadia_gui.

acadia_gui already embeds matplotlib in Qt (``LivePlotWidget`` uses
``FigureCanvasQTAgg`` + ``NavigationToolbar2QT``), so this reuses
:class:`~.interactive.SequenceView` for the behaviour and only adds the Qt
chrome. Nothing about the zoom logic is duplicated.

Integration into ``RightPanelTabs`` in ``acadia_gui/gui/main_data_browser.py`` is
two lines::

    from sequence_viz.qt_widget import SequenceWidget       # or acadia_qmsmt.sequence_viz
    self.sequence_tab = SequenceWidget()
    self.addTab(self.sequence_tab, "Pulse Sequence")

then, wherever the browser learns about the selected folder::

    self.sequence_tab.load_folder(path)

``load_folder`` is safe to call with any path -- non-data folders and trace
failures are reported in the widget instead of raising.

Importing this module requires PyQt5; the rest of sequence_viz does not.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure

from .folder import is_data_folder, trace_folder
from .interactive import SequenceView
from .plotting import fit_layout


class SequenceWidget(QWidget):
    """Show the compiled pulse sequence of a data folder, with drag-box zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trace = None
        self.view = None
        self.folder = None

        self.status = QLabel("Select a data folder to see its pulse sequence.")
        self.status.setWordWrap(True)

        self.point = QSpinBox()
        self.point.setRange(0, 0)
        self.point.setToolTip(
            "Sweep point. All points are captured by one dry run, so this "
            "switches instantly without re-tracing.")

        self.resolve = QSpinBox()
        self.resolve.setRange(0, 10_000_000)
        self.resolve.setSingleStep(100)
        self.resolve.setToolTip("Cycles to assume for register-driven lengths")

        self.saved_qmsmt = QCheckBox("saved qmsmt")
        self.saved_qmsmt.setChecked(True)
        self.saved_qmsmt.setToolTip(
            "Import against the acadia_qmsmt.py saved in the folder (falls back "
            "to the installed package if that fails)")

        self.group_copies = QCheckBox("group _copy pulses")
        self.group_copies.setChecked(True)

        self.envelopes = QCheckBox("envelopes")
        self.envelopes.setChecked(True)

        self.color_by = QComboBox()
        self.color_by.addItems(["memory", "name", "channel"])
        self.color_by.setToolTip(
            "memory: one hue per waveform memory — same hue means the same samples\n"
            "name:   one hue per pulse name, merged across channels\n"
            "channel: one hue per lane")

        self.envelope_source = QComboBox()
        self.envelope_source.addItems(["memory", "config"])
        self.envelope_source.setToolTip(
            "memory: the samples actually loaded at this sweep point\n"
            "config: the nominal pulse from the yaml")

        self.envelope_scale = QComboBox()
        self.envelope_scale.addItems(["per-pulse", "channel", "shared", "absolute"])
        self.envelope_scale.setToolTip(
            "per-pulse fills every bar (shape only); the others make amplitude "
            "comparable")

        self.envelope_mode = QComboBox()
        self.envelope_mode.addItems(["magnitude", "iq"])
        self.envelope_mode.setToolTip(
            "iq shows I and Q, the only way to see detune, phase and DRAG")

        self.jump = QComboBox()
        self.jump.setToolTip("Jump the viewport to a synchronizer block")
        self.jump.setMinimumWidth(160)

        self.reload_button = QPushButton("Reload")
        self.reset_button = QPushButton("Reset zoom")

        # top row: what to trace (needs a reload); bottom row: how to draw it
        controls = QHBoxLayout()
        controls.addWidget(QLabel("point"))
        controls.addWidget(self.point)
        controls.addWidget(QLabel("register cycles"))
        controls.addWidget(self.resolve)
        controls.addWidget(self.saved_qmsmt)
        controls.addStretch(1)
        controls.addWidget(QLabel("block"))
        controls.addWidget(self.jump)
        controls.addWidget(self.reload_button)
        controls.addWidget(self.reset_button)

        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("colour by"))
        view_controls.addWidget(self.color_by)
        view_controls.addWidget(self.envelopes)
        view_controls.addWidget(QLabel("from"))
        view_controls.addWidget(self.envelope_source)
        view_controls.addWidget(QLabel("scale"))
        view_controls.addWidget(self.envelope_scale)
        view_controls.addWidget(QLabel("mode"))
        view_controls.addWidget(self.envelope_mode)
        view_controls.addWidget(self.group_copies)
        view_controls.addStretch(1)

        self.canvas = FigureCanvasQTAgg(Figure())
        # matplotlib only delivers key events to a canvas that can take focus
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(view_controls)
        layout.addWidget(self.status)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.reload_button.clicked.connect(self.reload)
        self.reset_button.clicked.connect(self._reset)
        self.jump.currentIndexChanged.connect(self._jump_to_block)
        self.point.valueChanged.connect(self._select_point)
        for widget in (self.group_copies, self.envelopes):
            widget.stateChanged.connect(self._redraw)
        for combo in (self.color_by, self.envelope_source, self.envelope_scale,
                      self.envelope_mode):
            combo.currentIndexChanged.connect(self._redraw)

    # ---------------- public API ----------------

    def load_folder(self, folder):
        """Trace ``folder`` and display it. Never raises."""
        self.folder = str(folder) if folder else None
        if not self.folder or not is_data_folder(self.folder):
            self.clear("Not a deployed data folder "
                       "(needs kwargs.json and runtime.py).")
            return
        self.reload()

    def reload(self):
        if not self.folder:
            return
        self.status.setText(f"Tracing {self.folder} …")
        self.status.repaint()
        try:
            self.trace = trace_folder(
                self.folder,
                point=0,
                resolve_indeterminate=self.resolve.value(),
                use_saved_qmsmt=self.saved_qmsmt.isChecked())
        except Exception as exc:
            self.clear(f"Could not trace this folder: {type(exc).__name__}: {exc}")
            return
        self.point.blockSignals(True)
        self.point.setRange(0, max(self.trace.n_points - 1, 0))
        self.point.setValue(0)
        self.point.blockSignals(False)
        self._populate_blocks()
        self._redraw()
        self.status.setText(self.trace.summary())

    def _select_point(self, index):
        """Switch sweep point in place -- every point came from the one dry run."""
        if self.trace is None or self.view is None:
            return
        if not 0 <= index < self.trace.n_points:
            return
        self.view.set_point(index)
        self._populate_blocks()
        self.status.setText(self.trace.summary())

    def clear(self, message=""):
        self.trace = None
        if self.view is not None:
            self.view.disconnect()
            self.view = None
        self.canvas.figure.clear()
        self.canvas.draw_idle()
        self.jump.clear()
        self.status.setText(message)

    # ---------------- internals ----------------

    def _redraw(self):
        if self.trace is None:
            return
        keep = self.view.xlim_ns if self.view is not None else None
        if self.view is not None:
            self.view.disconnect()
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        self.view = SequenceView(
            self.trace, ax,
            group_copies=self.group_copies.isChecked(),
            show_envelopes=self.envelopes.isChecked(),
            color_by=self.color_by.currentText(),
            envelope_source=self.envelope_source.currentText(),
            envelope_scale=self.envelope_scale.currentText(),
            envelope_mode=self.envelope_mode.currentText())
        if keep is not None:
            self.view.set_window(*keep)
        fit_layout(figure, ax)
        self.canvas.draw_idle()

    def _populate_blocks(self):
        self.jump.blockSignals(True)
        self.jump.clear()
        self.jump.addItem("— whole sequence —", None)
        for blk in self.trace.blocks:
            self.jump.addItem(
                f"{blk.index}: {blk.start * self.trace.ns_per_cycle:.0f} ns",
                blk.index)
        self.jump.blockSignals(False)

    def _jump_to_block(self, _index):
        if self.trace is None or self.view is None:
            return
        which = self.jump.currentData()
        if which is None:
            self.view.reset()
            return
        blk = self.trace.blocks[which]
        ns = self.trace.ns_per_cycle
        pad = max(blk.length * ns * 0.15, 20.0)
        self.view.set_window(blk.start * ns - pad, blk.stop * ns + pad)

    def _reset(self):
        if self.view is not None:
            self.view.reset()
