""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

# get imports
import sys
import os
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib

# if modules folder, set to sys.path
func_path, _ = os.path.split(os.path.realpath(sys.argv[0]))
main_path, _ = os.path.split(func_path)
pack_path = os.path.join(main_path,'packages')
if os.path.isdir(pack_path):
    if pack_path not in sys.path:
        # add to sys.path
        sys.path.append(pack_path)
# import detectors
import detectors as det

def find_outliers(data_path):
    """ Print filenames and outlier indices for images in `data_directory`.

    Print filenames and detected outlier indices to the terminal.

    Parameters
    ----------
    data_path : str
        Directory containing containing images or path to specific file.
    plot : boolean
        optional choice to plot the result figures for each scan (default does not)
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

        # check for rms dvars outliers
        dvars, tmp = det.dvars_detector(data)
        outliers.extend(tmp)

        # check for outliers using PCA
        X_bad, tmp = det.pca_detector(data)
        outliers.extend(tmp)

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
    l_keys = sorted(list(data_dict.keys()))
    for key in l_keys:
        print(os.path.basename(key), ', '.join( repr(e) for e in data_dict[key]))

if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
