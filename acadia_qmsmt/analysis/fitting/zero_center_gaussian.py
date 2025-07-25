import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase

class ZeroCenterGaussian(FitterBase):
    @staticmethod
    def model(coordinates, A,  sigma, of):
        """$ A * exp((x/sigma)**2) + of $"""
        return A * np.exp(-(coordinates) ** 2 / (sigma ** 2)) + of

    @staticmethod
    def guess(coordinates, data):
        non_nan_data = data[np.isfinite(data)]
        of = (non_nan_data[0] + non_nan_data[-1]) / 2
        peak_idx = np.nanargmax(np.abs(data - of))
        A = data[peak_idx] - of
        return dict(A=A,  sigma=0.1, of=of)