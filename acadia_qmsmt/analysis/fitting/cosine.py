import matplotlib.pyplot as plt
import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase


class Cosine(FitterBase):
    @staticmethod
    def model(coordinates, A, f, phi, of):
        """$A cos(2 pi f x + phi) + of$"""
        return A * np.cos(2 * np.pi * coordinates * f + phi) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.

        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])

        return dict(A=A, f=f, phi=phi, of=of)

