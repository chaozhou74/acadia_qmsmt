import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt

from acadia_qmsmt.analysis import ComplexDataPointsType

TwoPi = 2 * np.pi

def rotate_iq(iq_pts:ComplexDataPointsType, angle: float = None):
    """
    Rotate the iq data points on the complex plane.

    :param iq_pts: Array of complex iq data points
    :param angle: rotation angle in radian. When not provided, will search for the angle that
        minimizes the variation of the q component.
    :return:
    """
    iq_pts = np.array(iq_pts)
    if angle is None:
        def std_q(theta_):
            iq_temp = iq_pts * np.exp(1j * theta_)
            return np.std(iq_temp.imag)

        res = minimize_scalar(std_q, bounds=[0, TwoPi])
        angle = res.x

    iq_new = iq_pts * np.exp(1j * angle)

    return iq_new

def cut_peak(data, cut_factor=0.5, plot=True, debug=False):
    """
    find the highest peak of a given dataset, set the region around the peak to np.nan. Returns the
    new data, and the left and right index of the peak. The peak region is set by cut_factor, which
    is in the unit of peak height.
    :param data: 1d array like data
    :param cut_factor: the height at which to cut the peak, in unit of peak height.
    :param plot: When true, plot the old data and the data after cutting.
    :return:
    """
    non_nan_data = data[np.isfinite(data)]
    off = (non_nan_data[0] + non_nan_data[-1]) / 2
    # find the highest peak
    peak0_idx = np.nanargmax(np.abs(data - off))
    peak0_y = data[peak0_idx]

    # cut peak
    y_span = peak0_y - off
    y_cut = peak0_y - y_span * cut_factor
    if debug:
        errstate_ = {}
    else:
        errstate_ = {"invalid": 'ignore'}
    with np.errstate(**errstate_):
        if peak0_y > off:
            cut_idx = np.where(data < y_cut)
        else:
            cut_idx = np.where(data > y_cut)

    # cut index to the right and left of peak0
    cut_idx_r = int(np.clip(np.where(cut_idx - peak0_idx > 0, cut_idx, np.inf).min(), 0, len(data)-1))
    cut_idx_l = int(np.clip(np.where(peak0_idx - cut_idx > 0, cut_idx, -np.inf).max(), 0, len(data)-1))

    # set peak region to nan
    temp_ = np.arange(0, len(data))
    new_data = np.where((temp_ > cut_idx_l) & (temp_ < cut_idx_r), np.nan, data)

    if plot:
        plt.figure()
        plt.plot(data)
        plt.plot(new_data)
        plt.plot([0, len(data)], [y_cut, y_cut], "--", color="k")

    return new_data, cut_idx_l, cut_idx_r