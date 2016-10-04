""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
def find_outliers(data_directory):
    """ Print filenames and outlier indices for images in `data_directory`.

    Print filenames and detected outlier indices to the terminal.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.

    Returns
    -------
    None
    """
    path_cwd = os.getcwd()
    path = os.path.join(path_cwd,data_directory)
    os.chdir(path)
    img = nib.load('group01_sub01_run1.nii', mmap=False)
    data = img.get_data()
    t = data.shape[-1]   #time points
    N = len(data[...,0].ravel()) #The number of voxels per volume

    print("Please select th)e method for outlier detection.")
    print("1: statistical (parametric) test")
    print("2: distance (non-parametric) based approach")
    print("3: Density (Cluster) based approach")
    print("4: High demiension adapted approach")
    print("5: outlier by standard variation")
    method = input(">")
    print('Your selected method is ', method)
    # print(type(method))
    if int(method) == 1:
        Mohalanobis_distance = []
        for i in range(t):
            # Z=data.shape[2]
            # for j in range(Z):
            #     print('Peiwu')
            X=data[:,:,:,i].ravel()
            unscaled_cov = X.dot(X.T) #This is covariance matrix for each volume
            # Mohalanobis distance for each multivariate data points i=1...., n is
            # denoted by Mi and given by:
            mean_4d = np.mean(data)
            X_mean_sub = X - mean_4d
            X_mean_unscaled_cov = X_mean_sub.T.dot(unscaled_cov)
            X_mean_cov = X_mean_unscaled_cov.dot(X_mean_sub)
            Mohalanobis_distance.append(np.sqrt(X_mean_cov))
        Mohalanobis_array = np.array(Mohalanobis_distance)
        # print("The shape of Mohalanobis_array is", Mohalanobis_array.shape)
        # The volume that has higher Mohalanobis index can be regarded as outliers
        max_i = np.argmax(Mohalanobis_array)
        outliers_array = Mohalanobis_array < Mohalanobis_array[max_i]
        plt.figure(1)
        plt.plot(Mohalanobis_array)
        plt.show()
        # print(outliers_array)
        data_remove_outliers = data[...,outliers_array]
        return data_remove_outliers

    elif int(method) == 2:
        print(method)
    elif int(method) == 3:
        print(method)
    elif int(method) == 4:
        print(method)
    elif int(method) == 5:
        vol_std = []
        for i in range(t):
            vol_std.append(data[...,i].std())
        plt.figure(2)
        plt.plot(np.array(vol_std))
        plt.show()
        #The std plot is very similar to Mohalanobis methos.
        min_i = np.argmin(np.array(vol_std))
        max_i = np.argmax(np.array(vol_std))
        outliers_array = (vol_std > vol_std[min_i])
        outliers_array[max_i] = False
        print(outliers_array)
        data_remove_outliers = data[...,outliers_array]
        return data_remove_outliers

    else:
        RuntimeError("please make a resonable choice")
    # Your code here
    # raise RuntimeError('No code yet')


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    find_outliers(data_directory)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
