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
import numpy.linalg as npl
import matplotlib.pyplot as plt

def afni_detector(data):
    """
    """
    # get number of timepoints and voxels per vol
    n_vols = data.shape[-1]
    N = np.prod(data.shape[:-1])
    # reshape the data
    data = data.reshape((N, n_vols))
    # calculate median voxel time series
    m_data = np.median(data, axis=1)
    # calculate clip level
    c = 0.5 * np.median(m_data)
    # ignore voxels with m_data < c
    data = data[m_data > c, :]
    m_data = m_data[m_data > c]
    m_data = np.outer(m_data, np.ones(161))
    # calculate Mean absolute deviation (MAD)
    MAD = np.median(np.abs(data - m_data), axis=1)
    # set upper and lower thresholds 5 mean absolute deviations
    u_thr = m_data + 5 * np.outer(MAD, np.ones(161))
    l_thr = m_data - 5 * np.outer(MAD, np.ones(161))
    # calculate outlying voxels from u_thr and l_thr
    n = np.sum(np.logical_or(data > u_thr, data < l_thr), axis=0)
    # find outlying timepoints
    outliers = [i for i, x in enumerate(n) if x > np.median(n) + 3.5 * np.median(MAD)]
    return n, outliers

def spm_global(vol):
    """
    """
    # calculate global mean for data
    thr = np.mean(vol) / 8
    # return mean value above thr
    return np.mean(vol[vol > thr])

def pca_detector(data):
    """
    """
    #- 'vol_shape' is the shape of volumes
    vol_shape = data.shape[:-1]
    #- 'n_vols' is the number of volumes
    n_vols = data.shape[-1]
    #- N is the number of voxels in a volume
    N = np.prod(vol_shape)

    #- Reshape to 2D array that is voxels by volumes (N x n_vols)
    X = data.reshape((N, n_vols)).T

    """
    The first part of the code will use PCA to get component matrix U
    and scalar projections matrix C
    """

    #- Calculate unscaled covariance matrix for X
    unscaled_covariance = X.dot(X.T)

    #- Use SVD to return U, S, VT matrices from unscaled covariance
    #U, S, VT = npl.svd(unscaled_covariance)
    U, S, VT = npl.svd(unscaled_covariance)

    #- Calculate the scalar projections for projecting X onto the vectors in U.
    #- Put the result into a new array C.
    C = U.T.dot(X)

    #- Transpose C
    #- Reshape the first dimension of C to have the 3D shape of the original data volumes.
    C_vols = C.T.reshape((vol_shape + (n_vols,)))

    """
    The second part of the code determines which voxels are inside the brain
    and which are outside the brain and creates a mask (boolean matrix)
    """

    #get the mean voxel intensity of entire 4D object
    mean_voxel = np.mean(data)
    #get the mean volume (3D) across time series (axis 3)
    mean_volume = np.mean(data, axis=3)
    #boolean mask set to all voxels above .5 in the first volume
    #(.125 is the SPM criterion but .5 seems like a better threshold)
    mask = mean_volume > (.5 * mean_voxel) #threshold can be adjusted!
    out_mask = ~mask

    """
    The third part of code finds the root mean square of U from step 1, then uses the
    mask from step 2 to determine which components explain data outside the brain
    Selects these "bad components" with high "outsideness"
    """

    #Apply mask to C matrix to get all voxels outside of brain
    outside = C_vols[out_mask]
    #Get RMS of the voxels outside, reflecting "outsideness" of this scan
    RMS_out = np.sqrt(np.mean((outside ** 2), axis=0))

    #Apply mask to C matrix to get all voxels inside brain
    inside = C_vols[mask]
    #Get RMS of the voxels inside, reflecting "insideness" of this scan
    RMS_in = np.sqrt(np.mean((inside ** 2), axis=0))

    #The closer this ratio is to 1, the worse the volume
    RMS_ratio = RMS_out / RMS_in

    """
    The fourth part of the code uses the "bad components" to generate a new
    "bad data set" and then puts this dataset through the outlier detector
    """

    #Create a boolean mask for the 10% worst PCs (meaning highest RMS ratio)
    PC_bad = np.percentile(RMS_ratio,90)
    PC_bad_mask = RMS_ratio > PC_bad

    U_bad = U[:, PC_bad_mask]
    C_bad = C[PC_bad_mask]

    #generates data set based on the bad PCs and (U and C matrices)
    X_bad = U_bad.dot(C_bad).T.reshape((vol_shape + (n_vols,)))

    # calculate outliers using iqr_detector
    _, outliers = mah_detector(X_bad)

    return X_bad, outliers

def pca_lev_detector(data):
    """
    """
    # get number of voxels per volume and n_vols
    n_vols = data.shape[-1]
    N = np.prod(data.shape[:-1])
    # reshape the data
    X = data.reshape((n_vols, N))
    # calculate mean of data at each voxel
    mean_vol = X.mean(axis=0)
    # subtract mean_vol from X to "center" the data
    X = X - mean_vol
    # calculate unscaled covariance
    unscaled_cov = X.dot(X.T)
    # reduce dimensionality with svd
    U, S, VT = npl.svd(unscaled_cov)
    # calculate the PCA leverage
    h = np.diag(U.dot(U.T))
    # calculate median of h
    # h_med = np.median(h)
    # # get outliers greater than 3 x median
    # outliers = [i for i, x in enumerate(h) if x > (3 * h_med)]
    _, outliers = iqr_detector(h)
    return h, outliers

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
    # thr = np.percentile(D, 90)
    # outliers = [i for i, x in enumerate(D) if x > thr]
    # return distances and outliers
    return D, outliers

def mean_detector(data):
    """
    """
    # init vol_mean
    vol_mean = []
    # for each timepoint
    for i in range(data.shape[-1]):
        # get spm global mean data
        vol_mean.append(spm_global(data[...,i]))
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
