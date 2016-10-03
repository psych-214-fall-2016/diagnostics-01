""" Python script to validate data

Run as:

    python3 scripts/validata_data.py data
"""

import os
import sys
import hashlib
path_cwd=os.getcwd()

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
    # path=path_cwd + '\\'+ data_directory
    # path=os.path.join(path_cwd,data_directory)
    # os.chdir(path)
    # print("Does the path conains the data?", os.path.exists('group01_sub01_run1.nii'))
    # Open the file, read contents as bytes.
    # Calculate, return SHA1 has on the bytes from the file.
    # n=len(os.listdir(os.getcwd())) -2 # there 2 extra filenames
    # hash_list=[]
    # for i in range(n/2):
    #     fname1='group01_sub0%i_run1'%i
    #     fname2='group01_sub0%i_run2'%i
    #     fobj1=open(fname1, 'rb')
    #     fobj2=open(fname2, 'rb')
    #     content1=fobj1.read()
    #     content2=fobj2.read()
    #     hash1=hashlib.sha1(content1).hexdigest()
    #     hash2=hashlib.sha1(contents),hexdigest()
    #     hash_list[i]=hash1 + fname1
    #     hash_list[i+1]=hash2+fname2
    #     i+=2
    fobj=open(filename,'rb')
    content=fobj.read()
    hash_data=hashlib.sha1(content).hexdigest()
    # raise RuntimeError('No code yet')
    # print(hash_list)
    # os.chdir(path_cwd)
    return hash_data

def validate_data(data_directory):
    # path_cwd=os.getcwd()
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
    """
    # Read lines from ``data_hashes.txt`` file.
    # Split into SHA1 hash and filename
    # Calculate actual hash for given filename.
    # If hash for filename is not the same as the one in the file, raise
    # ValueError
    path=os.path.join(path_cwd,data_directory)
    os.chdir(path)
    fobj=open('data_hashes.txt', 'rt')
    lines=fobj.readlines()
    for line in lines:
        # print(line)
        filename=line[-23:-1]
        hash_txt=line[0:-24]
        hash_list=file_hash(filename)
        print(hash_txt,hash_list)
        assert hash_txt == hash_list
    os.chdir(path_cwd)
    # raise RuntimeError("No code yet")


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    n=len(os.listdir(os.getcwd())) -2
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    path_cwd=os.getcwd()

    # Call function to validate data in data directory
    validate_data(data_directory)
    # path=path_cwd + '\\'+ data_directory

    os.chdir(path_cwd)

if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
