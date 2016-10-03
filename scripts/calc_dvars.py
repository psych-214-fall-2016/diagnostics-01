""" Script to calculate dvars values

For testing run::
    cd test
    python3 ..//scripts/calc_dvars.py small_4d.nii

You should see:

    [1.33811116,  1.53370821,  1.40858114]

If you make that work, try running this script over one of your own images.

Next, in IPython, import this module, and use the ``calc_image_dvars`` function
to calculate the DVARS values for one of your images.

Plot the results.

Don't forget the "%matplotlib" in IPython

If you make that work, check that the tests pass by:

* Installing the ``pytest`` module from the terminal with::

    pip install pytest

  Then run the tests with::

   py.test
"""

import sys
import os
import numpy as np

import nibabel as nib


def calc_image_dvars(img):
    """ Root mean squared difference between volumes in `img`.
    Parameters
    ----------
    img : image object
        nibabel image object containing 4D file, with last dimension length
        ``t``.

    Returns
    -------
    dvars : shape (t-1,) array
        1D array with root mean square difference values between each volume
        and the following volume
    """
    # For each voxel, calculate the differences between each volume and the one
    # following;
    #
    # Square the differences;
    # Sum over voxels for each volume, and divide by the number of voxels;
    # Return the square root of these values.
    data=img.get_data()
    print("The dimension of data is ", data.shape)
    t=data.shape[-1]
    vol0=data[...,0].ravel()
    N=len(vol0)
    shape=[]
    for i in range(t-1):
        distance=data[...,i] - data[...,i+1]
        d_sqrt=np.sum(distance**2)
        shape.append(np.sqrt(d_sqrt/N))
    # raise RuntimeError('No code yet')
    print('length of shape vecor is ', len(shape))
    return shape

def main():
    # Get the first command line argument

    filename = sys.argv[1]

    img = nib.load(filename, mmap=False)
    print(calc_image_dvars(img))


if __name__ == '__main__':
    main()
