#%%
%matplotlib qt

#%%
import numpy as np 
import matplotlib.pyplot as plt 
from utilsCoursework import dispImage
import skimage.io

#%%

def import_double_orientate(file):
    ''' 
    function to import, double, and reorientate and image
    PARAMS:
    INPUT:  file:   image
    OUTPUT: file:   image
    '''
    img = skimage.io.imread(file)
    file1 = np.double(img)
    # transpose - switch x and y dimensions
    #transpose_image = np.transpose(file1)
    # flip along second dimension - top to bottom
    #flipped_image = np.flip(transpose_image, 1)
    return file1

#%%

filepath = 'C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/test/'
image = 'test_1.png'
brain = 'test_1_BRAIN_STEM.png'
spine = 'test_1_SPINAL_CORD.png'

#%%

source_img = import_double_orientate(filepath + image)
    
brain_mask = import_double_orientate(filepath + brain)
mi, ma = np.floor(np.nanmin(brain_mask)), np.ceil(np.nanmax(brain_mask))
levels = np.arange(mi, ma+2, 2)
print(brain_mask.shape)

spine_mask = import_double_orientate(filepath + spine)
mi2, ma2 = np.floor(np.nanmin(spine_mask)), np.ceil(np.nanmax(spine_mask))
levels2 = np.arange(mi2, ma2+2, 2)
print(spine_mask.shape)

#%%

ax_im = plt.figure(figsize=(12,12))
ax_im, ax = plt.subplots()
ax.imshow(source_img, cmap='gray')
ax.contour(brain_mask, levels=levels, linewidths=1, colors=['black'])
ax.contour(spine_mask, levels=levels2, linewidths=1, colors=['black'])

#note to later indie, ax_im cant be shown by dispimage because it is a figure not an array
print(type(ax_im))
print(ax_im.shape)
#%%

array_img = np.asarray(ax_im)
print(type(array_img))
print(array_img.shape)
print(array_img)
print(type(ax))

# %%
dispImage(array_img)
# %%
dispImage(ax)
# %%
int_lims = [np.nanmin(array_img), np.nanmax(array_img)]
dispImage(array_img, int_lims)
# %%

import skimage.io
img = skimage.io.imread('C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/overlaid/tune_1.png')

# Check the data type of the image
print(img.dtype)

# convert data type to double to avoid errors when processing integers
import numpy as np

# convert img to double
img = np.double(img)
print(img.dtype)

print(img.shape)
print(img.size)

# SO to fix this you need to work out how to change the size of the image youve saved
# its currently sitting at 480,640,4

# %%
