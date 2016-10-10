"""get_data_hashes.py

Creates a .json file with hashes for the testing data ds114
and the filtered data ds005_filtered containing file names
and corresponding hashes.
Input the directories paths in dirs_ls

Run with:
   python get_data_hashes.py
   from the directory relative to the input paths

"""
from __future__ import print_function
from data_hashes import *
import json
import pdb

if __name__ == "__main__":
    dirs_ls = ['group01']
    for d in dirs_ls:
        d_dict = generate_dir_md5(d)
        with open(d + '_hashes.json', 'w') as file_out:

            json.dump(d_dict, file_out)
