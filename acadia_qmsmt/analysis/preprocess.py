from typing import Union
import math
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt

from acadia_qmsmt.analysis import ComplexDataPointsType

TwoPi = 2 * np.pi

def reshape_iq_data_by_axes(raw_data, *axes: Union[list, np.ndarray]):
    """
    Reshape IQ data (data with last axis of shape 2)  to match the shape defined by the sweep axes.
    Truncates the data to exclude any incomplete iterations.

    :param raw_data: np.ndarray, assumed to have shape (..., 2).
    :param axes: sweep axes, provided from outermost to innermost. Each axis should be a list or np.ndarray.

    :return : np.ndarray of shape (completed_iterations, *len(axes), 2), or None if not enough data.
    """

    ax_shapes = []
    for ax in axes:
        ax_shapes.append(len(ax))
    raw_data = raw_data.reshape(-1, 2)
    points_per_iter = math.prod(ax_shapes)
    completed_iterations = len(raw_data) // points_per_iter
    if completed_iterations == 0:
        return None

    valid_points = completed_iterations * points_per_iter
    data = raw_data[:valid_points, ...]
    data = data.reshape(completed_iterations, *ax_shapes, 2)

    return data


def find_iq_rotation(iq_pts:ComplexDataPointsType):
    """
    Find the rotation angle in radian that minimizes the 
    variation of the q component.

    :param iq_pts: Array of complex iq data points

    :return: optimal angle in radians
    """
    def std_q(theta_):
        iq_temp = iq_pts.ravel() * np.exp(1j * theta_)
        return np.std(iq_temp.imag)

    res = minimize_scalar(std_q, bounds=[0, TwoPi])
    angle = res.x
    
    return angle


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
        angle = find_iq_rotation(iq_pts)

    iq_new = iq_pts * np.exp(1j * angle)

    return iq_new

def to_complex(x, allow_1d_real: bool = False, flatten: bool = False) -> np.ndarray:
    """
    Convert various IQ input formats to a 1D complex numpy array.

    Accepted formats:
      - 1D complex array
      - ND array with shape (..., 2): [...,0] is real, [...,1] is imag
      - Optional: 1D real array, interpreted as real-only (if allow_1d_real=True)

    :param x: Input array (list, numpy array, etc).
    :param allow_1d_real: If True, allow 1D real inputs and interpret as purely real.
    :param flatten: If True, return a 1D flattened array.
    :return: ND array of complex numbers.
    :raises ValueError: If input is not a valid IQ format.
    """
    x = np.asarray(x)

    if np.iscomplexobj(x):
        return x.ravel() if flatten else x

    elif x.ndim >= 1 and x.shape[-1] == 2:
        iq = x[..., 0] + 1j * x[..., 1]
        return iq.ravel() if flatten else iq

    elif allow_1d_real and x.ndim == 1 and np.issubdtype(x.dtype, np.floating):
        iq = x.astype(np.complex128)
        return iq.ravel() if flatten else iq

    else:
        raise ValueError(f"Unsupported IQ format: shape={x.shape}, dtype={x.dtype}")

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