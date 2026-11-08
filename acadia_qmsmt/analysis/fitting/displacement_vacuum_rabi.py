import numpy as np
from scipy.stats import poisson

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase


class DisplacementVacuumRabi(FitterBase):
    """Qubit P(e) after D(A) on a mode in vacuum followed by one calibrated qubit-mode swap (pi on |1,g> <-> |0,e>):

        P_e(A) = offset + scale * sum_{n>=1} Poisson(n; (A * dac2alpha)^2) * sin^2(sqrt(n) * pi/2)

    dac2alpha is coherent-state amplitude per unit DAC scale; scale/offset absorb readout and preparation errors.
    The swap is assumed calibrated, so its angle is not a fit parameter.
    The automatic guess scans dac2alpha on a log grid, each candidate with its own linear scale/offset, so a
    fast oscillation cannot win on the ripple. Pass bounds on dac2alpha to lock the branch by hand.
    """

    N_MAX = 400

    @staticmethod
    def model(coordinates, dac2alpha, scale, offset):
        A = np.atleast_1d(np.asarray(coordinates, dtype=float))
        n = np.arange(1, DisplacementVacuumRabi.N_MAX)
        p_n = poisson.pmf(n[None, :], ((A * dac2alpha) ** 2)[:, None])
        return offset + scale * (p_n @ np.sin(np.sqrt(n) * np.pi / 2) ** 2)

    @staticmethod
    def guess(coordinates, data):
        A = np.asarray(coordinates, dtype=float)
        y = np.asarray(data, dtype=float)
        a_max = max(float(np.max(np.abs(A))), 1e-9)
        best = (np.inf, 1.0 / a_max, 1.0, 0.0)
        for k in np.logspace(-1, 1.5, 300) / a_max:
            m = DisplacementVacuumRabi.model(A, k, 1.0, 0.0)
            (s, o), *_ = np.linalg.lstsq(np.column_stack([m, np.ones_like(m)]), y, rcond=None)
            cost = float(np.sum((s * m + o - y) ** 2))
            if cost < best[0]:
                best = (cost, k, s, o)
        _, k, s, o = best
        return {"dac2alpha": {"value": float(k), "min": 1e-9}, "scale": float(s), "offset": float(o)}
