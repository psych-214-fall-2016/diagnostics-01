"""Diagnostics contains some functions used for outlier detection.
   Group 1: Joe, Just0n, Peiwu. 10/04/2016
"""
import numpy as np
def vol_std(data):
    """ Return standard deviation across voxels for 4D array data
    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes.  Call the shape
        of this array (M, N, P, T) where T is the number of volumes.
    Returns
    -------
    std_values : array shape (T,)
        One dimensonal array where ``std_values[i]`` gives the standard
        deviation of all voxels contained in ``data[..., i]``.
    """
    T = data.shape[-1]
    data_2d = data.reshape((-1, T))
    return np.std(data_2d, axis = 0)

def vol_rms_diff(data):
    """ Return root mean square of differences between sequential volumes
    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes.  Call the shape
        of this array (M, N, P, T) where T is the number of volumes.
    Returns
    -------
    rms_values : array shape (T-1,)
        One dimensonal array where ``rms_values[i]`` gives the square root of
        the mean (across voxels) of the squared difference between volume i and
        volume i + 1.
    """
    T=data.shape[-1]
    diff_data = np.diff(data, axis=-1)
    diff_data_2d = np.reshape(diff_data, (-1, T-1))
    return np.sqrt(np.mean(diff_data_2d ** 2, axis=0))

def extend_diff_outliers(diff_indices):
    """ Extend difference-based outlier indices `diff_indices` by pairing
    Parameters
    ----------
    diff_indices : array
        Array of indices of differences that have been detected as outliers.  A
        difference index of ``i`` refers to the difference between volume ``i``
        and volume ``i + 1``.
    Returns
    -------
    extended_indices : array
        Array where each index ``j`` in `diff_indices has been replaced by two
        indices, ``j`` and ``j+1``, unless ``j+1`` is present in
        ``diff_indices``.  For example, if the input was ``[3, 7, 8, 12, 20]``,
        ``[3, 4, 7, 8, 9, 12, 13, 20, 21]``. """

    # for each index
    for x in diff_indices:
        # include x + 1 if not already in diff_indices
        if not x + 1 in diff_indices:
            diff_indices = np.append(diff_indices, x + 1)
    # return sorted array of indices
    diff_indices.sort()
    return diff_indices

def iqr_outliers(data, iqr_scale=1.5):
    """ Return indices of outliers identified by interquartile range
    Parameters
    ----------
    data : 4D array
        One-dimensional numpy array, from which we will identify outlier
        values.
    iqr_scale : float, optional
        Scaling for IQR to set low and high thresholds.  Low threshold is given
        by 25th centile value minus ``iqr_scale * IQR``, and high threshold id
        given by 75 centile value plus ``iqr_scale * IQR``.
    Returns
    -------
    outlier_indices : array
        Array containing indices in `arr_2d` that contain outlier values.
    lo_hi_thresh : tuple
        Tuple containing 2 values (low threshold, high thresold) as described
        above.
    """

    T=data.shape[-1]
    data_2d = data.reshape((-1, T))
    data_25, data_75 = np.percentile(data_2d, [25, 75], axis = 0)
    IQR = np.mean(data_75 - data_25)
    Hi_threshold = np.mean(data_75) + IQR * iqr_scale
    Lo_threshold = np.mean(data_25) + IQR * iqr_scale
    outlier_indices = (data_2d > Hi_threshold) | (data_2d < Lo_threshold)
    lo_hi_thresh = (Lo_threshold, Hi_threshold)
    return outlier_indices, lo_hi_thresh
