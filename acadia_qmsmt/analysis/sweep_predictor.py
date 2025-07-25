from typing import Union, List
import numpy as np


def polyfit_predict(history_x, history_y, new_x, order=2, debug=False):
    """
    Fit a 1D polynomial to the provided history data and evaluate it at `new_x`.

    :param history_x: List or array of past x-values.
    :param history_y: List or array of past y-values corresponding to `history_x`.
    :param new_x: The x-value at which to predict the y-value.
    :param order: Degree of the polynomial to fit. Will be reduced if not enough points.
    :param debug: If True, also return the fitted polynomial coefficients.
    :return: Predicted y-value at `new_x`, or (y-value, coeffs) if debug is True.
    """
    history_x = np.asarray(history_x)
    history_y = np.asarray(history_y)

    deg = np.min([len(history_x) - 1, order])
    coeffs = np.polyfit(history_x, history_y, deg=deg)
    y_pred = np.polyval(coeffs, new_x)
    if not debug:
        return y_pred
    else:
        return y_pred, coeffs


class PolyPredictor:
    """
    Class for maintaining a stream of (x, y) data and predicting future y-values
    using polynomial extrapolation over a moving window.
    """

    def __init__(self, window_size: int = 3, poly_order: int = 2):
        """
        Initialize the predictor.

        :param window_size: Number of recent points to use for each polynomial fit.
            By default, will try to use window_size//2 number of points on the right, and the rest on the left.
            Will be reduced if not enough points.
        :param poly_order: Maximum degree of the polynomial to fit.
        """
        self.window_size = window_size
        self.poly_order = poly_order
        self.x_vals = []
        self.y_vals = []
        self.fig, self.plot_ax = None, None

    def observe_and_predict(self, x_val, y_val, new_x, debug=False, plot=False, plot_ax=None):
        """
        Add a new (x, y) observation and predict the y value at `new_x`.

        :param x_val: New x-value to add to the history.
        :param y_val: New y-value to add to the history.
        :param new_x: x-value at which to predict the next y.
        :param debug: :param debug: When True, return the fitted poly coefficients and x, y data for ploy fitting as well
        :return: Predicted y-value at `new_x`.
        """
        self.observe(x_val, y_val)

        # Always get debug info if plotting is requested
        if debug or plot:
            new_y, coeffs, x_data, y_data = self.predict(new_x, debug=True)
            if plot:
                self.plot_prediction(x_val, y_val, new_x, new_y, coeffs, x_data, plot_ax)
            return (new_y, coeffs, x_data, y_data) if debug else new_y

        return self.predict(new_x)

    def observe(self, x_val, y_val):
        """
        Add a single (x, y) observation to the internal history.

        :param x_val: x-value of the observation.
        :param y_val: y-value of the observation.
        """
        self.x_vals.append(x_val)
        self.y_vals.append(y_val)
        self.x_vals, self.y_vals = map(list, zip(*sorted(zip(self.x_vals, self.y_vals))))

    def predict(self, new_x, debug=False):
        """
        Predict the y-value at a given `new_x` using polynomial extrapolation.

        :param new_x: x-value at which to make the prediction.
        :param debug: When True, return the fitted poly coefficients and x, y data for ploy fitting as well

        :return: Predicted y-value at `new_x`.
        """
        n = len(self.x_vals)
        if n == 0:
            raise ValueError("No data points to predict from.")
        if n <= self.window_size:
            x_data = self.x_vals
            y_data = self.y_vals
        else:
            center = np.searchsorted(self.x_vals, new_x)
            half = self.window_size // 2

            # Clamp window to fit in range
            start = max(0, min(n - self.window_size, center - half))
            end = start + self.window_size

            x_data = self.x_vals[start:end]
            y_data = self.y_vals[start:end]

        if debug:
            return *polyfit_predict(x_data, y_data, new_x, self.poly_order, debug=debug), x_data, y_data

        return polyfit_predict(x_data, y_data, new_x, self.poly_order, debug=debug)

    def plot_prediction(self, x_obs, y_obs, x_pred, y_pred, coeffs, x_fit, plot_ax=None):
        """
        Plot the observed point, fitted curve, and predicted value.

        :param x_obs: The observed x value just added.
        :param y_obs: The observed y value just added.
        :param x_pred: The x value to predict at (usually next time step).
        :param y_pred: The predicted y value.
        :param coeffs: Fitted polynomial coefficients.
        :param x_fit: x-values used for fitting.
        :param plot_ax: Optional matplotlib axis to plot on. If None, creates one.
        """
        import matplotlib.pyplot as plt

        if plot_ax is None:
            if self.plot_ax is None:
                self.fig, self.plot_ax = plt.subplots()
            plot_ax = self.plot_ax
        else:
            self.fig, self.plot_ax = plot_ax.figure, plot_ax

        idx = len(self.x_vals)

        plot_ax.scatter(x_obs, y_obs, marker="*", color=f"C{idx}", s=100)

        x_fit_fine = np.linspace(min(min(x_fit), x_pred), max(max(x_fit), x_pred), 50)
        plot_ax.plot(x_fit_fine, np.polyval(coeffs, x_fit_fine), color=f"C{idx + 1}")
        plot_ax.scatter(x_pred, y_pred, marker="o", edgecolors=f"C{idx + 1}", facecolors="none", s=150)
        plt.pause(0.1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_data = np.concatenate([np.linspace(2, 0, 5), np.linspace(0.61, 1.61, 5)])
    y_data = np.cos(x_data)

    pp = PolyPredictor(window_size=4, poly_order=3)

    fig, ax = plt.subplots(1, 1)
    for idx, x in enumerate(x_data):
        y = np.cos(x)
        if idx < len(x_data)-1:
            new_y, coeffs, x_fit, y_fit = pp.observe_and_predict(x, y, x_data[idx + 1], debug=True, plot=True)

