"""data.py

Simple script to validate the downloaded with
with the hashes in a .json file.

Prerequesite :
   To create the json file relative to a specific
   folder, see project-epsilon/data/get_data_hashes.py

Assuming the ds005_hashes.json file is present
in the directory:
   python data.py
"""

import pdb
import json

from data_hashes import check_hashes

if __name__ == "__main__":
    file_ls = ['group01_hashes.json']
    for f in file_ls:
        with open(f) as infile:
            f_dict = json.load(infile)
            check_hashes(f_dict)
