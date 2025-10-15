import matplotlib.pyplot as plt
import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase


class ExponentialKappa(FitterBase):
    @staticmethod
    def model(coordinates, A, kappa, of):
        """ A * exp (-x*kappa) + of"""
        return A * np.exp(-coordinates * kappa) + of

    @staticmethod
    def guess(coordinates, data):
        of = data[-1]
        A = data[0] - data[-1]
        kappa_ = 4./(coordinates[-1] - coordinates[0])
        kappa = {"value": kappa_, "min": 1e-10}
        return dict(A=A, kappa=kappa, of=of)