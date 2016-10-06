#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
# Display array values to 6 digits of precision
np.set_printoptions(precision=4, suppress=True)

#: import numpy.linalg with a shorter name
import numpy.linalg as npl

#- Change this section to load any image
import nibabel
img = nibabel.load('data/group01_sub01_run1.nii')
#- Get the data array from the image
data = img.get_data()

#- Make variables:
#- 'vol_shape' for shape of volumes
vol_shape = data.shape[:-1]
#- 'n_vols' for number of volumes
n_vols = data.shape[-1]
#- N is the number of voxels in a volume
N = np.prod(vol_shape)

"""
The first part of the code will use PCA to get component matrix U
and scalar projections matrix C
"""

#- Reshape to 2D array that is voxels by volumes (N x n_vols)
X = data.reshape((N, n_vols))

#- Calculate unscaled covariance matrix for X
unscaled_covariance = X.dot(X.T) """why does this break python?"""

#- Use SVD to return U, S, VT matrices from unscaled covariance
U, S, VT = npl.svd(unscaled_covariance)

#- Calculate the scalar projections for projecting X onto the vectors in U.
#- Put the result into a new array C.
C = U.T.dot(X)

#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the original data volumes.
C_vols = C.T.reshape(vol_shape, n_vols) """check this - not sure this works"""

"""
The second part of the code determines which voxels are inside the brain
and which are outside the brain and creates a mask (boolean matrix)
"""

#boolean mask set to all voxels above .2 in the first volume
volume_one = data[..., 0]
vol_mean = np.mean(volume_one)
mask = volume_one > (.2*vol_mean) """what is the best threshold?"""
outside_brain = data[~mask] """put this here or wait for C matrix?"""

"""
The third part of code finds the root mean square of U from step 1, then uses the
mask from step 2 to determine which components explain data outside the brain
Selects these "bad components" with high "outsideness"
"""

#Take first volume from 4D C matrix
C_vol0 = C[..., 0]
#Get all voxels that are outside of brain
voxels_outside = c_vol0[~mask]
#Get RMS of the C matrix , reflecting intensity of the component outside the brain
RMS = np.sqrt(np.sum(voxels_outside ** 2))

"""
The fourth part of the code uses the "bad components" to generate a new
"bad data set" and then puts this dataset through the outlier detector
"""
