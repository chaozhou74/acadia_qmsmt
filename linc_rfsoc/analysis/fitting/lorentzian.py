import matplotlib.pyplot as plt
import numpy as np

from linc_rfsoc.analysis.fitting.fitter_base import FitterBase
from .preprocess import cut_peak


class Lorentzian(FitterBase):
    @staticmethod
    def model(coordinates, A, x0, k, of):
        """$ A /(k*(x-x0)**2+1) +of $"""
        return A / (k * (coordinates - x0) ** 2 + 1) + of

    @staticmethod
    def guess(coordinates, data):
        non_nan_data = data[np.isfinite(data)]
        of = (non_nan_data[0] + non_nan_data[-1]) / 2
        peak_idx = np.nanargmax(np.abs(data - of))
        x0 = coordinates[peak_idx]
        A = data[peak_idx] - of
        new_data, cut_idx_l, cut_idx_r = cut_peak(data, plot=False)
        half_peak_idx = cut_idx_r
        half_peak_width_2 = coordinates[half_peak_idx]-x0 if half_peak_idx!=peak_idx else coordinates[peak_idx + 1] - x0
        k = 1 / (half_peak_width_2) ** 2
        return dict(A=A, x0=x0, k=k, of=of)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = None
        ax.plot(self.coordinates, self.data, ".")
        fine_coords = np.linspace(self.coordinates[0], self.coordinates[-1], len(self.coordinates)*10)
        ax.plot(fine_coords, self.result.eval(coordinates=fine_coords))

        return fig, ax

