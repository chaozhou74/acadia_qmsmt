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
        of = (np.max(data) + np.min(data)) / 2
        A = (np.max(data) - np.min(data)) / 2.

        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])
        phi = np.pi

        # robast phase estimation
        omega = 2 * np.pi * f * coordinates
        cos_part = np.cos(omega)
        sin_part = np.sin(omega)
        B, C = np.linalg.lstsq(np.stack([cos_part, sin_part], axis=1), data - of, rcond=None)[0]
        phi = np.arctan2(-C, B)  # minus sign to convert sin to cos form

        # Final amplitude might be refined here:
        A = np.hypot(B, C)

        return dict(A=A, f=f, phi=phi, of=of)

