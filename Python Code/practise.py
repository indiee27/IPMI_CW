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

tune_list = ['tune_1_','tune_2_','tune_3_']
atlas_list = ['atlas_1_','atlas_2_','atlas_3_','atlas_4_','atlas_5_']
test_list = ['test_1_','test_2_','test_3_','test_4_','test_5_']

#%%

#from demonsReg import demonsReg

n = 0

tune_name = str(tune_list[n])
t_o_name = tune_name + 'overlaid.png'
tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
dispImage(tune_overlay)

#import the mask and source
t_m_name = tune_name + 'mask.png'
tune_mask = skimage.io.imread(t_m_name, as_gray=True)
tune_mask[tune_mask<1] = 0
    
t_s_name = tune_name + 'source.png'
tune_source = skimage.io.imread(t_s_name, as_gray=True)

#%%
atlas_name = str(atlas_list[n])
a_s_name = atlas_name + 'source.png'
atlas_source = skimage.io.imread(a_s_name, as_gray=True)

#%%
from demonsReg import demonsReg

#%%

img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it=30)


# %%

warped_name = tune_name + atlas_name + 'warped.png'
result_name = tune_name + atlas_name + 'result.png'
print(warped_name)
print(result_name)
#%%

#save the figures
#plt.figure()
ax,plot = plt.subplots()
plot.set_axis_off()
ax.add_axes(plot)
plot.imshow(img_warped, cmap='gray')
plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)

#%%

a_m_name = atlas_name + 'mask.png'
atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
atlas_mask[atlas_mask<1] = 0
atlas_mask = np.flip(atlas_mask, 0)

#%%

from utilsCoursework import resampImageWithDefField
import numpy.ma as ma
#warp contours
warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

# mask resulting image
result = ma.masked_where(warped_atlas_mask == 0, img_warped)
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

# %%
img_test = skimage.io.imread('overlaid_images/atlas_1_source.png', as_gray=True)

# %%
plt.imshow(img_test)
plt.savefig('param_1/test.png')
# %%
