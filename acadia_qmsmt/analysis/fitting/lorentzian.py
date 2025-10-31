import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase
from acadia_qmsmt.analysis.preprocess import cut_peak


class Lorentzian(FitterBase):
    @staticmethod
    def model(coordinates, A, x0, k, of):
        """$ A /(k*(x-x0)**2+1) +of $"""
        return A / (k * (coordinates - x0) ** 2 + 1) + of

    @staticmethod
    def guess(coordinates, data):
        non_nan_data = data[np.isfinite(data)]
        of = np.mean(non_nan_data)
        peak_idx = np.nanargmax(np.abs(data - of))
        x0 = coordinates[peak_idx]
        A = data[peak_idx] - of
        new_data, cut_idx_l, cut_idx_r = cut_peak(data, plot=False)
        half_peak_idx = cut_idx_r
        half_peak_width_2 = coordinates[half_peak_idx]-x0 if half_peak_idx!=peak_idx else coordinates[peak_idx + 1] - x0
        k = 1 / (half_peak_width_2) ** 2

        max_mag = np.max(data) - np.min(data)
        A = {"value": A, "min": -max_mag*2, "max": max_mag*2}
        x0 = {"value": x0, "min": np.min(coordinates), "max":np.max(coordinates)}


        return dict(A=A, x0=x0, k=k, of=of)

