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
from PIL import Image
def is_binary(file):
    img = file
    w,h = img.shape
    for i in range(w):
        for j in range(h):
            x = (i, j)
            if x == 1 or x == 0:
                pass
            else:
                IOError

#%%
n = 0
msd_results = []
reg_weight_results = []
brain_warp_results = []
spine_warp_results = []

while n < len(atlas_list):
    # import atlas ct image
    atlas_name = str(atlas_list[n])
    atlas_file_loc = 'overlaid_images/' + atlas_name
    a_s_name = atlas_file_loc + 'source.png'
    atlas_source = skimage.io.imread(a_s_name, as_gray=True)
       
    # registration
    img_warped, img_def = demonsReg(atlas_source, target_source, disp_freq=0, num_lev=1, max_it=10)

    # import spinal cord and brain stem binary images
    brain = skimage.io.imread('masks/' + atlas_name + 'brain.png', as_gray=True)
    spine = skimage.io.imread('masks/' + atlas_name + 'spine.png', as_gray=True)

    #make sure its binary values
    brain[brain>0.5] = 1
    brain[brain<0.5] = 0
    spine[spine>0.5] = 1
    spine[spine<0.5] = 0

    # numbered names
    brain_name = 'brain_warp_' + str([n])
    spine_name = 'spine_warp_' + str([n])

    # warp binary images
    brain_name = resampImageWithDefField(brain, img_def)
    spine_name = resampImageWithDefField(spine, img_def)

    # check warps are binary
    is_binary(brain_name)
    is_binary(spine_name)

    #calculate lmsd between target and warped
    msd = calcLMSD(target_source, img_warped, 20)

    w,h = msd.shape
    registration_weight = np.ndarray([w,h])
    for i in range(w):
        for j in range(h):
            x = msd[i][j]
            x_reg = 1/(1+x)
            registration_weight[i][j] = x_reg

    msd_results.append(msd)
    reg_weight_results.append(registration_weight)
    brain_warp_results.append(brain_warp)
    spine_warp_results.append(spine_warp)
    
    n = n + 1


# %%
for n in range(4):
    w,h = msd.shape
    msd_new = np.ndarray([w,h])
    for i in range(w):
        for j in range(h):
            x = msd[i][j]
            x_reg = 1/(1+x)
            msd_new[i][j] = x_reg
# %%
print(msd_new)
# %%
print(msd.shape)
print(msd_new.shape)
# %%
w,h = target_source.shape
sum_registration_weights = np.ndarray([w,h])
for a in range(4):
    sum_registration_weights[i][j] = np.sum(reg_weight_results[a])

# %%
print(sum_registration_weights.shape)
# %%
print(sum_registration_weights)
# %%
print(sum_registration_weights[0][0][0])
# %%
print(sum_registration_weights[0][0])
# %%
w,h = target_source.shape
a = len(atlas_list)
reg_weight_norm = np.ndarray([a,w,h])
for a in range(len(atlas_list)):
    item = reg_weight_results[a]
    for i in range(w):
        for j in range(h):
            x = item[i][j]
            y = sum_registration_weights[i][j]
            value = x/y
    reg_weight_norm[a][i][j] = value
# %%
print(reg_weight_norm.shape)
# %%
print(reg_weight_norm[0][0][0])
# %%
print(reg_weight_norm[1])
# %%
MAS_prob = np.ndarray([w,h])
mas_list = []
for a in range(len(atlas_list)):
    mas = np.multiply(reg_weight_norm[a], brain_warp_results[a])
    mas_list.append(mas)

# %%
print(len(mas_list))

# %%
for a in range(len(atlas_list)):
    MAS_prob[i][j] = np.sum(mas_list[a])
# %%
print(MAS_prob)
# %%
plt.imshow(MAS_prob)
# %%
