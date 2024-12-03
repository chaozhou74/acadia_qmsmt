import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase
from acadia_qmsmt.analysis.fitting.preprocess import cut_peak


class Gaussian(FitterBase):
    @staticmethod
    def model(coordinates, A, x0, sigma, of):
        """$ A /(k*(x-x0)**2+1) +of $"""
        return A * np.exp(-(coordinates - x0) ** 2 / (2 * sigma ** 2)) + of

    @staticmethod
    def guess(coordinates, data):
        non_nan_data = data[np.isfinite(data)]
        of = (non_nan_data[0] + non_nan_data[-1]) / 2
        peak_idx = np.nanargmax(np.abs(data - of))
        x0 = coordinates[peak_idx]
        A = data[peak_idx] - of
        new_data, cut_idx_l, cut_idx_r = cut_peak(data, cut_factor=np.exp(-0.5), plot=False)
        half_peak_idx = cut_idx_r
        sigma = coordinates[half_peak_idx]-x0 if half_peak_idx!=peak_idx else coordinates[peak_idx + 1] - x0
        return dict(A=A, x0=x0, sigma=sigma, of=of)