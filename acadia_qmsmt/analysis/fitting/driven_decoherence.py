import matplotlib.pyplot as plt
import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase

class DrivenDecoherence(FitterBase):
    @staticmethod
    def model(coordinates, A, f, phi, kappa_1, kappa_phi, of) -> np.ndarray:
        return A/2. * np.exp(-coordinates *kappa_1) *(1 + np.exp(-coordinates *kappa_phi)*np.cos(f * np.pi * 2 * coordinates + phi) ) + of

    @staticmethod
    def guess(coordinates, data):
        of = 0.
        A = (np.max(data) - np.min(data))
        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])
        tau_1 = (1 / 4.0) * (coordinates[-1] - coordinates[0])
        tau_phi = (1 / 2.0) * (coordinates[-1] - coordinates[0])
        kappa_1 = {"value": 1./tau_1, "bounds": (1e-10, np.inf)}
        kappa_phi = {"value": 1./tau_phi, "bounds": (1e-10, np.inf)}

        return dict(A=A, f=f, phi=phi, kappa_1=kappa_1,kappa_phi=kappa_phi, of=of)