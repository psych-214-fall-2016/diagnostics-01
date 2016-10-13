"""

"""
from __future__ import division, print_function
import sys, os, pdb
import numpy as np
import nibabel as nib

# def plot_mosaic(img_data, transpose=False):
def plot_mosaic(img_data):
    """ Return a mosaic plot for each slice of
        the 3rd dimension of img_data

    Parameters:
    ----------
    img_data = 3D array

    Returns:
    -------
    grid_2D : a 2D image with each slice of
        the 3rd dimension of img_data plotted
	in a mosaic
    """
    n_slices = img_data.shape[-1]
    # Dimensions of the mosaic grid
    n_rows = int(np.ceil(float(np.sqrt(n_slices))))
    n_cols = int(np.ceil(float(n_slices)/float(n_rows)))
    # Define the 2D mosaic
    grid_2D = np.zeros((n_rows*img_data.shape[0], n_cols*img_data.shape[1]))
    z = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if z < n_slices:
                img_data_slice = img_data[:,::-1,z]
                grid_2D[i*img_data.shape[0]:(i+1)*img_data.shape[0], j*img_data.shape[1]:(j+1)*img_data.shape[1]] = img_data_slice
                z += 1
    return grid_2D
