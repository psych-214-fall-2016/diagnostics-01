"""data_hashes.py

A collection of functions manages hases to data download.
See test_* functions in this directory for the nose tests.  

"""

from __future__ import print_function, division
import hashlib
import os
import pdb
import json

def generate_file_md5(filename, blocksize=2**20):
    """Generate the md5 hash for filename
    Parameters
    ---------
    filename: the file to genererate the hash for
    Returns
    -------
    m.hexdigest: concatenated strings of the md5-file
        generated for filename
    """
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
#            buf.decode('latin-1').encode('utf-8')
            if not buf:
                break
            m.update(buf)
#        f.close()
    return m.hexdigest()

def generate_dir_md5(dirname):
    """Generate the md5 hash for each file in dirname
    Parameters
    ---------
    dirname: folder path to locate the files to genererate
       the hashes for
    Returns
    -------
    dir_hash: json file with hashes of each file in dirname
    """
    dir_hash = {}
    for root, dirs, files in os.walk(dirname):
        for f in files:
           file_p = os.path.join(root, f)
           dir_hash[file_p] = generate_file_md5(file_p)
    return dir_hash


def check_hashes(newDict):
    """Check if the generated hashes match the reference in newDict
    Parameters
    ---------
    newDict: Reference dictionary containing file names in keys and
        correct corresponding hashes as values
    Returns
    -------
    all_good: bool
        True if all generated hashes correcpond to the stores hashes
        in newDict
    """
    all_good = True
    for k, v in newDict.items():
        digest = generate_file_md5(k)
        if v == digest:
            print("The file {0} has the correct hash.".format(k))
        else:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
    return all_good
