#%%
%matplotlib auto

import numpy as np 
import matplotlib.pyplot as plt 
from utilsCoursework import dispImage
import skimage.io

filepath_name = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/test/")

img = skimage.io.imread(filepath_name + 'test_1.png')

def orientation_standard(img):
    """ reorientate image to standard orientation """
    # transpose - switch x and y dimensions
    transpose_image = np.transpose(img)
    # flip along second dimension - top to bottom
    flipped_image = np.flip(transpose_image, 1)
    return flipped_image

img2 = orientation_standard(img)

plt.figure(1)
dispImage(img2)
# %%

#mask import
brain_mask = skimage.io.imread(filepath_name + 'test_1_BRAIN_STEM.png')
spine_mask = skimage.io.imread(filepath_name + 'test_1_SPINAL_CORD.png')

#orientate
brain = np.flip(brain_mask,0)
spine = np.flip(spine_mask,0)

#make contour
fig = plt.figure(2)
dispImage(img2)
brain_contour = plt.contour(brain, colors='black')
spine_contour = plt.contour(spine, colors='black')

#how do I save this as an image? It has to be able to use shape
print(fig)

# %%
