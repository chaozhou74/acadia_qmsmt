from typing import Union, List, Tuple, Literal

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label
from numpy.typing import NDArray

StateCircleType = Tuple[complex, np.floating]  # (I_center + 1j * Q_center, circle_radius)
ComplexDataPointsType = Union[List[complex], np.ndarray[complex]]
ComplexDataTracesType = Union[List, np.ndarray]

QubitStateLabels=["g", "e", "f", "h", "i"]

def find_quadrant(z: Union[complex, NDArray[complex]]):
    """
    find the quadrant number of a complex number z

    :param z: (Array of) complex number
    :return: quadrant value, [1,2,3,4]
    """
    quadrant = np.array([3, 2, 4, 1])[2 * (z.real > 0) + (z.imag > 0)]
    return quadrant


def quadrant_signs(quadrant:Union[int, NDArray[int]]) -> NDArray[int]:
    """
    Returns the signs of the real and imaginary parts of a complex number in the given quadrant.

    :param quadrant: quadrant number [1,2,3,4]
    :return:  (sign of real, sing of imag) represented by +1/-1 s
    """
    if not (set(np.array([quadrant]).flatten()) <= {1, 2, 3, 4}):
        raise ValueError("Quadrant must be one of [1, 2, 3, 4].")
    quadrant_to_signs = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)])
    signs = quadrant_to_signs[quadrant-1]
    return signs


def population_in_quadrant(iq_pts: ComplexDataPointsType, quadrant: Literal[1,2,3,4] = None,
                           i_threshold: int = 0, q_threshold: int = 0, axis:int=0):
    """
    Calculate the population of IQ data points (`iq_pts`) that fall into a specific quadrant
    of the IQ plane.

    :param iq_pts: Complex IQ data points.
    :param quadrant: The specific quadrant to calculate the population for.
    :param i_threshold: I value for the vertical line separating left and right quadrants.
    :param q_threshold: Q value for the horizontal line separating left and right quadrants.
    :param axis: The iteration axis of the data. 
        e.g. For a spectroscopy data in shape (iterations, n_freqs), axis should be 0, and the
        return will be in shape (n_freqs)
    :return:
    """

    pts_shifted = iq_pts - i_threshold - 1j * q_threshold
    pct = np.mean(find_quadrant(pts_shifted) == quadrant, axis=axis)

    return pct


def mask_state_with_circle(iq_pts: ComplexDataPointsType, state_circle: StateCircleType) -> List[bool]:
    """
    Generate a mask for the input IQ data points that are within the `state_circle`

    :param iq_pts: input IQ data points, should be an array of complex numbers

    :return:
    """
    state_center, state_r = state_circle
    mask = abs(iq_pts - state_center) < state_r

    return mask


def _hist2d_with_indices(iq_pts: ComplexDataPointsType, **kwargs):
    """
    Compute a 2D histogram of IQ data points, as well as the bin indices of where each data point falls in the grid.

    :param iq_pts: List of complex IQ points
    :param kwargs: kwargs for `np.histogram2d`
    :return:
    """
    hist, xedges, yedges = np.histogram2d(iq_pts.real, iq_pts.imag, **kwargs)
    bin_indices = np.digitize(iq_pts.real, xedges[:-1]) - 1, np.digitize(iq_pts.imag, yedges[:-1]) - 1

    return hist, xedges, yedges, bin_indices


def find_most_significant_blob(iq_pts: ComplexDataPointsType,
                               bins: int = 50, sigma_factor: float = 2.5) -> StateCircleType:
    """
    Identify the most significant Gaussian blob in an 1D array of complex IQ points.
    Different blobs are separated by looking at bin connectivity.

    No fitting is performed here, so hopefully this is fast and robust...

    :param iq_pts: 1D array of complex numbers representing IQ points.
    :param bins: Number of bins for the 2D histogram.
    :param sigma_factor: The factor for the radius (e.g., 2 for 2-sigma).
    :return: A tuple (center, radius) for the most significant blob.
    """
    # Create a 2D histogram
    hist, xedges, yedges, bin_indices = _hist2d_with_indices(iq_pts, bins=bins)

    # Find the connected components in the histogram
    structure = np.ones((3, 3))  # Connectivity structure for 2D neighbors
    labeled, num_features = label(hist > 0, structure=structure)

    # Find the connected region with the largest total hist count (most significant blob)
    blob_weights = [hist[labeled == i].sum() for i in range(1, num_features + 1)]
    most_significant_blob_idx = np.argmax(blob_weights) + 1

    # Mask for points in the largest blob
    largest_blob_mask = (labeled == most_significant_blob_idx)

    # Extract points in the largest blob by mapping histogram bins back to points
    in_largest_blob = largest_blob_mask[bin_indices]

    blob_points = iq_pts[in_largest_blob]
    blob_pt_bin_indices = bin_indices[0][in_largest_blob], bin_indices[1][in_largest_blob]
    pt_weights = hist[blob_pt_bin_indices]

    # Compute the center and radius
    center = np.average(blob_points, weights=pt_weights)
    distances = np.abs(blob_points - center)
    radius = sigma_factor * np.std(distances)

    return center, radius



def find_state_circles(*state_pts: ComplexDataPointsType, bins: int = 50,
                       sigma_factor: float = 2.5, average_radius=True, debug=False) -> List[StateCircleType]:
    """
    Find the state circle for an arbitrary number of prepared state data points.

    The raw data from each prepared state will be masked based on its prominence comparing to data from other sates
    before looking for the state circle.

    :param state_pts: Variable number of lists of complex IQ data points for each prepared state.
    :param bins: Number of bins for the 2D histogram.
    :param sigma_factor: The sigma factor for the radius (e.g., 2 for 2-sigma).
    :param average_radius: When True, average the radius of all state circles
    :return:
    """
    center_list = []
    r_list = []
    n_states = len(state_pts)

    # make individual histograms for each state using the same grid, for prominence comparison
    hist_list = []
    bin_indices_list = []
    all_pts = np.concatenate(state_pts).flatten()
    compare_hist_bin = 41  # this needs to be relatively small for the data masking to work
    real_bins = np.linspace(all_pts.real.min(), all_pts.real.max(), compare_hist_bin)
    imag_bins = np.linspace(all_pts.imag.min(), all_pts.imag.max(), compare_hist_bin)

    for i, pts in enumerate(state_pts):
        hist, _, _, bin_indices = _hist2d_with_indices(pts, bins=[real_bins, imag_bins])
        hist_list.append(hist / len(pts))
        bin_indices_list.append(bin_indices)

    hist_all = np.sum(hist_list, axis=0)

    # find circle for each state
    if debug:
        fig, axs = plt.subplots(len(state_pts), 1, figsize=(4, n_states * 3))
        axs = [axs] if n_states == 1 else axs
        axs[0].set_title("find_state_circles debug")

    for i, pts in enumerate(state_pts):
        # make a hist of the current state
        hist, bin_indices = hist_list[i], bin_indices_list[i]

        # mask data of the current state by comparing the hist of current data and the rest data
        hist_diff = 2 * hist - hist_all
        prominent_region = (hist_diff > 0)  # region in the 2D hist where current data is higher than the rest
        data_mask = prominent_region[bin_indices]  # mask data points in that region
        pts_vld = pts[data_mask]

        # find state circle using the valid data
        center, radius = find_most_significant_blob(pts_vld, bins, sigma_factor)
        center_list.append(center)
        r_list.append(radius)

        if debug:
            axs[i].hist2d(pts_vld.real, pts_vld.imag, bins=bins, cmap="hot")
            axs[i].add_patch(patches.Circle((center.real, center.imag), radius, edgecolor="w", facecolor="none"))
            axs[i].set_aspect(1)

    if average_radius:
        r_list = [np.mean(r_list)] * n_states

    state_circles = [(c, r) for c, r in zip(center_list, r_list)]

    return state_circles