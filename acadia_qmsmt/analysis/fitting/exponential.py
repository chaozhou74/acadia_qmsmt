import matplotlib.pyplot as plt
import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase


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
        tau = {"value": tau_, "bounds": (1e-10, np.inf)}
        return dict(A=A, tau=tau, of=of)

    def plot(self, ax=None):
        fig, ax = super().plot(ax)
        ax.set_title(f"tau: {self.ufloat_results['tau']}")