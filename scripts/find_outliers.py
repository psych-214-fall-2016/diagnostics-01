""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

# get imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# if modules folder, set to sys.path
if os.path.isdir('modules'):
    if os.path.abspath('modules') not in sys.path:
        # add to sys.path
        sys.path.append(os.path.abspath('modules'))
# import detectors
import detectors as det

def find_outliers(data_path):
    """ Print filenames and outlier indices for images in `data_directory`.

    Print filenames and detected outlier indices to the terminal.

    Parameters
    ----------
    data_path : str
        Directory containing containing images or path to specific file.

    Returns
    -------
    outliers : data dictionary
        Dictionary with each file and its corresponding outliers
    """

    # init data_dict
    data_dict = {}
    # ensure data_path is fullpath
    data_path = os.path.abspath(data_path)
    # check if data_path is directory or file
    if os.path.isdir(data_path):
        # return nii files
        data_files = []
        for f in os.listdir(data_path):
            if f[-4:] == '.nii':
                data_files.append(os.path.join(data_path,f))
    elif os.path.isfile(data_path):
        # set data_files to [data_path]
        data_files = [data_path]
    else:
        # error
        raise('Directory does not exist or unknown file')

    # for each fileName, load image and find outliers
    for fileName in data_files:
        # get img and data
        img = nib.load(fileName, mmap=False)
        data = img.get_data()

        # init outliers
        outliers = []
        # check for mean outliers
        vol_mean, tmp = det.mean_detector(data)
        outliers.extend(tmp)
        # check for std outliers
        vol_std, tmp = det.std_detector(data)
        outliers.extend(tmp)
        # check for Mahalanobis outliers
        D, tmp = det.mah_detector(data)
        outliers.extend(tmp)
        # check for rms dvars outliers
        dvars, tmp = det.dvars_detector(data)
        outliers.extend(tmp)
        # PCA?

        # get unique outliers
        outliers = list(set(outliers))
        # set data_dict
        data_dict[fileName] = sorted(outliers)

    # return the data dictionary with outliers
    return data_dict

def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    data_dict = find_outliers(data_directory)

    # print data dictionary of fileNames and outliers
    for key in data_dict:
        print(os.path.basename(key), data_dict[key])

if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
