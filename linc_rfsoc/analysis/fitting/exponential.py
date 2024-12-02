import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter

from linc_rfsoc.analysis.fitting.fitter_base import FitterBase


class Exponential(FitterBase):
    @staticmethod
    def model(coordinates, A, tau, of):
        """ A * exp (-x/tau) + of"""
        return A * np.exp(-coordinates / tau) + of

    @staticmethod
    def guess(coordinates, data):
        of = data[-1]
        A = data[0] - data[-1]
        tau_ = (coordinates[-1] - coordinates[0])/3
        tau = Parameter("tau", value=tau_,  min=0.0000001)
        return dict(A=A, tau=tau, of=of)

    def plot(self, ax=None):
        fig, ax = super().plot(ax)
        ax.set_title(f"tau: {self.ufloat_results['tau']}")