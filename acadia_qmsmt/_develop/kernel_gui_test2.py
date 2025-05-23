import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# might be going too far on this...

class InteractiveCircle:
    def __init__(self, circle, update_callback):
        self.circle = circle
        self.update_callback = update_callback
        self.press = None
        self.resizing = False
        self.connect()

    def connect(self):
        self.circle.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.circle.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.circle.figure.canvas.mpl_connect("button_release_event", self.on_release)

    def on_press(self, event):
        if event.xdata is None or event.ydata is None:
            return
        contains, _ = self.circle.contains(event)
        if contains:
            distance = np.sqrt(
                (event.xdata - self.circle.center[0]) ** 2
                + (event.ydata - self.circle.center[1]) ** 2
            )
            # Check if near the edge of the circle for resizing
            if abs(distance - self.circle.radius) < 0.05:
                self.resizing = True
            else:
                self.press = (self.circle.center, (event.xdata, event.ydata))

    def on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self.resizing:
            dx = event.xdata - self.circle.center[0]
            dy = event.ydata - self.circle.center[1]
            new_radius = np.sqrt(dx**2 + dy**2)
            self.circle.set_radius(new_radius)
            self.update_callback(self.circle)
        elif self.press is not None:
            center, (xpress, ypress) = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            new_center = (center[0] + dx, center[1] + dy)
            self.circle.set_center(new_center)
            self.update_callback(self.circle)

    def on_release(self, event):
        self.press = None
        self.resizing = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Histogram with Interactive Circles")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)

        # Create a matplotlib figure for the 2D histogram
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        # Generate sample complex data
        self.data = self.generate_complex_data()

        # Plot the initial histogram and circles
        self.plot_histogram()

    def generate_complex_data(self):
        np.random.seed(0)
        g_blobs = 0.2 * np.random.randn(500) + 1 + 1j * (0.2 * np.random.randn(500) + 1)
        e_blobs = 0.2 * np.random.randn(500) - 1 + 1j * (0.2 * np.random.randn(500) - 1)
        return np.concatenate([g_blobs, e_blobs])

    def plot_histogram(self):
        ax = self.figure.add_subplot(111)
        ax.clear()

        # Generate 2D histogram data
        x = self.data.real
        y = self.data.imag
        ax.hist2d(x, y, bins=50, cmap="Blues")

        # Add initial circles
        self.g_circle = Circle((1, 1), 0.3, color="red", fill=False, lw=2)
        self.e_circle = Circle((-1, -1), 0.3, color="green", fill=False, lw=2)
        ax.add_patch(self.g_circle)
        ax.add_patch(self.e_circle)

        # Make circles interactive
        self.g_interactive = InteractiveCircle(self.g_circle, self.update_circle)
        self.e_interactive = InteractiveCircle(self.e_circle, self.update_circle)

        self.g_position = self.g_circle.center
        self.e_position = self.e_circle.center

        self.canvas.draw()

        ax.set_aspect(1)

    def update_circle(self, circle):
        if circle == self.g_circle:
            self.g_position = circle.center
        elif circle == self.e_circle:
            self.e_position = circle.center
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
