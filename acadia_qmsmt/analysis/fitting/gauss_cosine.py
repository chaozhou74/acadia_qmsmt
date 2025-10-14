import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase

class GaussCosine(FitterBase):
    @staticmethod
    def model(coordinates, A, f, phi, tau, of) -> np.ndarray:
        """ A * cos(2 pi f x + phi) * exp (-(x/tau)**2) + of"""
        return A * np.cos(f * np.pi * 2 * coordinates + phi) * np.exp(-(coordinates / tau)**2) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.
        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])
        tau_ = (1 / 4.0) * (coordinates[-1] - coordinates[0])
        tau = {"value":tau_, "bounds":(1e-10, coordinates[-1]*20)}
        return dict(A=A, f=f, phi=phi, tau=tau, of=of)