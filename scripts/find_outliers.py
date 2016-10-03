""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

import sys
import os
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
    path_cwd=os.getcwd()
    path=os.path.join(path_cwd,data_directory)
    os.chdir(path)
    print("Please select the method for outlier detection.")
    print("1: statistical (parametric) test")
    print("2: distance (non-parametric) based approach")
    print("3: Density based approach")
    print("4: High demiension adapted approach")
    method=input(">")
    
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
