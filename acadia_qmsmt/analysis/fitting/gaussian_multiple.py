import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase
from acadia_qmsmt.analysis.preprocess import cut_peak


class DoubleGaussian(FitterBase):
    @staticmethod
    def model(coordinates, A0, x0, sigma0,A1,x1,sigma1, of):
        """$ A /(k*(x-x0)**2+1) +of $"""
        return A0 * np.exp(-(coordinates - x0) ** 2 / (2 * sigma0 ** 2)) + A1 * np.exp(-(coordinates - x1) ** 2 / (2 * sigma1 ** 2)) + of

    @staticmethod
    def guess(coordinates, data):
        non_nan_data = data[np.isfinite(data)]
        of = (non_nan_data[0] + non_nan_data[-1]) / 2
        peak_idx = np.nanargmax(np.abs(data - of))

        # from gemini
        from scipy.signal import find_peaks
        # Find all peaks and calculate their properties (including prominence)
        # Prominence is the height of the peak relative to the surrounding "valley"
        peak_indices, properties = find_peaks(data)

        # Get the prominence values for all found peaks
        prominences = properties['prominences']

        # Get the indices that would sort 'prominences' in descending order
        sorted_indices_of_peaks = np.argsort(prominences)[::-1]

        # Select the original indices (from the y_vals array) of the top 2 peaks
        top_two_indices = peak_indices[sorted_indices_of_peaks[:2]]

        # Get the x-locations of the top two peaks
        peak_locations_x = coordinates[top_two_indices]
        peak_locations_x.sort() # Sort to present them in ascending order

        x0 = coordinates[peak_idx]
        A = data[peak_idx] - of
        
        new_data, cut_idx_l, cut_idx_r = cut_peak(data, cut_factor=np.exp(-0.5), plot=False)
        half_peak_idx = cut_idx_r
        sigma = coordinates[half_peak_idx]-x0 if half_peak_idx!=peak_idx else coordinates[peak_idx + 1] - x0
        return dict(A0=A,A1=A, x0=peak_locations_x[0],x1=peak_locations_x[1], sigma0=sigma,sigma1=sigma, of=of)