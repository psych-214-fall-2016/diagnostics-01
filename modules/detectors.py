""" Utilities for detecting outliers

These functions take a vector of values, and return a boolean vector of the
same length as the input, where True indicates the corresponding value is an
outlier.

The outlier detection routines will likely be adapted to the specific measure
that is being worked on.  So, some detector functions will work on values > 0,
other on normally distributed values etc.  The routines should check that their
requirements are met and raise an error otherwise.
"""

# Python 2 compatibility
from __future__ import print_function, division

# Any imports you need
import numpy as np

def dvars_detector(data):
    """
    """
    # init dvars
    dvars = []
    # for 0 to timepoints-1
    for i in range(data.shape[-1] - 1):
        # get difference between volumes
        vol_diff = data[...,i + 1] - data[..., i]
        # calculate rms
        dvars.append(np.sqrt(np.mean(vol_diff ** 2)))
    # get outliers using iqr_detector
    _, outliers = iqr_detector(dvars)
    # return rms dvars and outliers
    return dvars, outliers

def mah_detector(data):
    """ Detect outliers using Mahalanobis distance:

    """
    # init distance
    D = []
    # get mean of the 4d data
    M = np.mean(data, axis=3).ravel()
    # for each timepoint
    for i in range(data.shape[-1]):
        # get data for volume
        X = data[...,i].ravel()
        # get covariance of the data
        S = X.dot(X.T)
        # calculate the Mahalanobis distance for the volume
        D.append(np.sqrt((X - M).T.dot(S ** -1).dot((X - M))))
    # get outliers using iqr_detector
    _, outliers = iqr_detector(D)
    # return distances and outliers
    return D, outliers

def mean_detector(data):
    """
    """
    # init vol_mean
    vol_mean = []
    # for each timepoint
    for i in range(data.shape[-1]):
        # get mean data
        vol_mean.append(data[...,i].mean())
    # get outliers using iqr_detector
    _, outliers = iqr_detector(vol_mean)
    # return volume means and outliers
    return vol_mean, outliers

def std_detector(data):
    """
    """
    # init volume standard deviations
    vol_std = []
    # for each timepoint
    for i in range(data.shape[-1]):
        # append standard deviation
        vol_std.append(data[...,i].std())
    # get outliers using iqr_detector
    _, outliers = iqr_detector(vol_std)
    # return volume standard deviations and outliers
    return vol_std, outliers

def iqr_detector(measures, iqr_proportion=1.5):
    """ Detect outliers in `measures` using interquartile range.

    Returns a boolean vector of same length as `measures`, where True means the
    corresponding value in `measures` is an outlier.

    Call Q1, Q2 and Q3 the 25th, 50th and 75th percentiles of `measures`.

    The interquartile range (IQR) is Q3 - Q1.

    An outlier is any value in `measures` that is either:

    * > Q3 + IQR * `iqr_proportion` or
    * < Q1 - IQR * `iqr_proportion`.

    See: https://en.wikipedia.org/wiki/Interquartile_range

    Parameters
    ----------
    measures : 1D array
        Values for which we will detect outliers
    iqr_proportion : float, optional
        Scalar to multiply the IQR to form upper and lower threshold (see
        above).  Default is 1.5.

    Returns
    -------
    outlier_tf : 1D boolean array
        A boolean vector of same length as `measures`, where True means the
        corresponding value in `measures` is an outlier.
    """
    # improt numpy
    import numpy as np

    # calculate 25, 50, 75 percentiles of measures
    Q1, Q3 = np.percentile(measures, [25,75])

    # calculate interquartile range
    IQR = Q3 - Q1

    # calculate upper and lower outlier values
    upper_out = Q3 + IQR * iqr_proportion
    lower_out = Q1 - IQR * iqr_proportion

    # get outliers based on > upper_out or < lower_out
    outlier_tf = np.logical_or(measures > upper_out, measures < lower_out)

    # get indices of outlier_tf
    outlier_i = [i for i, x in enumerate(outlier_tf) if x]

    return outlier_tf, outlier_i
