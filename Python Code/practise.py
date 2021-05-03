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

tune_list = ['tune_1_','tune_2_','tune_3_']
atlas_list = ['atlas_1_','atlas_2_','atlas_3_','atlas_4_','atlas_5_']
test_list = ['test_1_','test_2_','test_3_','test_4_','test_5_']

# %%
#section 1.3 trialling
target_name = 'test_1_'
target_file_loc = 'overlaid_images/' + target_name
t_s_name = target_file_loc + 'source.png'
target_source = skimage.io.imread(t_s_name, as_gray=True)

# %%
from demonsReg import demonsReg
from utilsCoursework import resampImageWithDefField, calcLMSD

#%%
n = 0

atlas_name = str(atlas_list[n])
atlas_file_loc = 'overlaid_images/' + atlas_name
a_s_name = atlas_file_loc + 'source.png'
atlas_source = skimage.io.imread(a_s_name, as_gray=True)
print(atlas_source.shape)
print(target_source.shape)
#%%
img_warped, img_def = demonsReg(atlas_source, target_source, disp_freq=0, max_it=10, num_lev=1)

print(img_warped.shape)
print(img_def.shape)
#%%
# import spinal cord and brain stem binary images
brain = skimage.io.imread('atlas_1_brain.png', as_gray=True)
spine = skimage.io.imread('atlas_1_spine.png', as_gray=True)

#%%
# names
brain_warp_name = atlas_name + 'brain_warp.png'
spine_warp_name = atlas_name + 'spine_warp.png'

#%%
# warp binary images
brain_warp = resampImageWithDefField(brain, img_def)
brain_warp[brain_warp>0.5] = 1
brain_warp[brain_warp<0.5] = 0
plt.imshow(brain_warp_corrected)
plt.axis('off')
plt.savefig(brain_warp_name, bbox_inches='tight', pad_inches=0)
spine_warp = resampImageWithDefField(spine, img_def)
spine_warp[spine_warp>0.5] = 1
spine_warp[spine_warp<0.5] = 0
plt.imshow(spine_warp, cmap='gray')
plt.axis('off')
plt.savefig(spine_warp_name, bbox_inches='tight', pad_inches=0)
    
msd = []
msd = calcLMSD(target_source, img_warped, 20)

# calculated lmsd


#print(msd[0])
# %%

