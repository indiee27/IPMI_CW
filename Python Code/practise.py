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
total_mask = spine_mask + brain_mask

#%%
print(total_mask.shape)

#%%
#plots = plt.figure(frameon=False)
plots,ax = plt.subplots()
ax.set_axis_off()
plots.add_axes(ax)
ax.contour(total_mask, linewidths=1, colors=['black'])
plt.savefig('mask.png', bbox_inches='tight', pad_inches=0)

#%%
maskmask = skimage.io.imread('mask.png', as_gray=True)
dispImage(maskmask)
#%%
maxi = np.amin(maskmask)
print(maxi)
#%%
#mask = plt.figure()

mask1,mask = plt.subplots()
mask.contour(brain_mask, levels=levels, linewidths=1, colors=['black'])
mask.contour(spine_mask, levels=levels2, linewidths=1, colors=['black'])
plt.savefig('mask.png')

#%%
mask2[mask2<255] = 0
mask2[mask2==255] = 1

#%%

