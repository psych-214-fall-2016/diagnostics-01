""" Python script to validate data

Run as:

    python3 scripts/validata_data.py data
"""

import os
import sys
import hashlib

def file_hash(filename):
    """ Get byte contents of file `filename`, return SHA1 hash

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    hash : str
        SHA1 hexadecimal hash string for contents of `filename`.
    """
    # Open the file, read contents as bytes.
    fobj = open(filename, 'rb')
    # get data, close fobj
    data = fobj.read()
    fobj.close()
    # Calculate, return SHA1 has on the bytes from the file.
    hash = hashlib.sha1(data).hexdigest()
    return hash


def validate_data(data_directory):
    """ Read ``data_hashes.txt`` file in `data_directory`, check hashes

    Parameters
    ----------
    data_directory : str
        Directory containing data and ``data_hashes.txt`` file.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If hash value for any file is different from hash value recorded in
        ``data_hashes.txt`` file.
        Otherwise prints "[filename] has correct hash"
    """
    # Read lines from ``data_hashes.txt`` file.
    fobj = open(os.path.join(data_directory, 'data_hashes.txt'),'rt')
    lines = fobj.readlines()
    # Split into SHA1 hash and filename
    sha1s = []
    filenames = []
    for line in lines:
        # split line
        [tmp1, tmp2] = line.split()
        # set sha1 and filename
        sha1s.append(tmp1)
        filenames.append(tmp2)
    # Calculate actual hash for given filename.
    for i in range(len(filenames)):
        # calculate hash
        hash = file_hash(os.path.join(data_directory, filenames[i]))
        # If hash for filename is not the same as the one in the file, raise
        # ValueError
        if hash != sha1s[i]:
            raise ValueError
        else:
            print(filenames[i] + " has correct hash")

def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    validate_data(data_directory)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
