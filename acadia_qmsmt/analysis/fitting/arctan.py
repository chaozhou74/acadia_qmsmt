import numpy as np

from acadia_qmsmt.analysis.fitting.fitter_base import FitterBase
from scipy.ndimage import gaussian_filter1d


class Arctan(FitterBase):
    @staticmethod
    def model(coordinates, x0, A, w, of):
        return A * np.arctan((coordinates - x0)/w) + of

    @staticmethod
    def guess(coordinates, data):
        smoothed_data = gaussian_filter1d(data, sigma=3)
        dphi = np.gradient(smoothed_data, coordinates)
        x0_guess = coordinates[np.argmax(np.abs(dphi))]
        scale_guess = np.mean(data[-3:] - data[:3])/np.pi
        w_guess = (coordinates[-1] - coordinates[0]) * 0.02
        offset_guess = np.median(smoothed_data[:5]) + scale_guess * np.pi/2

        if scale_guess < 0:
            scale_min, scale_max = scale_guess*2 ,0
        else:
            scale_min, scale_max = 0, scale_guess*2
        return {
            "x0": {"value": x0_guess, "min": coordinates[0], "max": coordinates[-1]},
            "A": {"value": scale_guess, "min": scale_min, "max": scale_max},
            "w": {"value": w_guess, "min": coordinates[1] - coordinates[0], "max": coordinates[-1] - coordinates[0]},
            "of": {"value": offset_guess},
        }




class ArctanTilt(FitterBase):
    # todo: This actually doesn't work well. When the slope is large, 
    # the residual of the arctan's central feature gets buried under the dominant linear background.
    @staticmethod
    def model(coordinates, x0, A, w, of, slope):
        return A * np.arctan((coordinates - x0)/w) + of + slope * (coordinates - coordinates[0])

    @staticmethod
    def guess(coordinates, data):

        # take the average of beginning and ending slopes for e_delay guess
        k_fit_idx =  np.max([len(coordinates)//10, 4])
        k0, _ = np.polyfit(coordinates[:k_fit_idx], data[:k_fit_idx], deg=1)
        k1, _ = np.polyfit(coordinates[-k_fit_idx:], data[-k_fit_idx:], deg=1)
        slope_guess = (k0+k1) / 2

        # use same guess as a normal arctan with the slope part removed
        arctan_guesses = Arctan.guess(coordinates,  data - slope_guess * (coordinates - coordinates[0]))
        arctan_guesses["slope"] = {"value": slope_guess, "min": slope_guess- abs(slope_guess)*0.001,
                                   "max":slope_guess + abs(slope_guess)*0.001}
        return arctan_guesses

