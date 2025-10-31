import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase

class Linear(FitterBase):
    @staticmethod
    def model(coordinates, m, b) -> np.ndarray:
        """ y = m*x + b"""
        return m*coordinates + b

    @staticmethod
    def guess(coordinates, data):
        b = np.mean(data)
        m = (np.max(data) - np.min(data)) / (np.max(coordinates) - np.min(coordinates))
        return dict(m=m, b=b)