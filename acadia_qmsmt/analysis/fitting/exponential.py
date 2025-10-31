import matplotlib.pyplot as plt
import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase


class Exponential(FitterBase):
    @staticmethod
    def model(coordinates, A, tau, of):
        """ A * exp (-x/tau) + of"""
        return A * np.exp(-coordinates / tau) + of

    @staticmethod
    def guess(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)

        # Offset: median of last 20% (min 3 pts)
        tail = max(3, len(y)//5)
        of = float(np.median(y[-tail:]))

        # Amplitude from first point
        A = float(y[0] - of)
        if A == 0:  # tiny nudge to avoid degenerate logs
            A = 1e-6

        # Tau: time to reach |A|/e from the start (no logs)
        x0 = float(x[0]); y0 = float(y[0])
        target = of + np.sign(A) * (abs(A) / np.e)
        idx = np.where((y - target) * (y0 - target) <= 0)[0]  # first crossing
        if idx.size > 0 and idx[0] > 0:
            i = idx[0] - 1
            # linear interpolation for crossing
            t = (target - y[i]) / (y[i+1] - y[i] + 1e-12)
            x_e = x[i] + t * (x[i+1] - x[i])
            tau = float(max((x_e - x0), 1e-10))
        else:
            # simple fallback if no crossing in data span
            tau = float(max((x[-1] - x[0]) / 5.0, 1e-10))
        return dict(A=float(A), tau=float(tau), of=float(of))
