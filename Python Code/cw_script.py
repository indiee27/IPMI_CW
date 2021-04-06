""" 
Image registration and segmentation script for IPMI Coursework
Indie Lewis Thompson
UCL
"""
#%%

# imports
import numpy as np 
import numpy.ma as ma
import matplotlib.pyplot as plt 
import skimage.io
#from PIL import ImageOps
import os
from utilsCoursework import dispImage

%matplotlib qt

#%%

# load in all the data from the atlas and tuning patients
atlas_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/atlas/")
tune_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/tune/")
test_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/test/")

def get_imlist(path):
    """ returns a list of filenames for all images in a directory """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

atlas_data = get_imlist(atlas_data_str)
tune_data = get_imlist(tune_data_str)
test_data = get_imlist(test_data_str)

# reorientate to standard orientation

def orientation_standard(img):
    """ reorientate image to standard orientation """
    # transpose - switch x and y dimensions
    transpose_image = np.transpose(img)
    # flip along second dimension - top to bottom
    flipped_image = np.flip(transpose_image, 1)
    return flipped_image

# convert all images to double and reorientate

def double_img(img):
    """ returns a double array """
    return np.double(img)

# function to overlay images with double and reorientate

def image_import_overlay(filepath, image_brain, image_spine, image):
    """ 
    returns image overlaid with contour of image brain and image spine 
    reorientate image to standard orientation and double the input
    
    INPUTS:     filepath: either atlas_data, tune_data or test_data
                image_brain: brain stem image name
                image_spine: spinal cord image name
                image: ct scan image name

    OUTPUTS:    source_img: the image returned with contoured overlays
    """

    # make the mask
    mask_brain_read = skimage.io.imread(filepath + image_brain)
    mask_spine_read = skimage.io.imread(filepath + image_spine)

    mask_brain_double = double_img(mask_brain_read)
    mask_spine_double = double_img(mask_spine_read)

    mask_brain = np.flip(mask_brain_double,0)
    mask_spine = np.flip(mask_spine_double,0)

    # import the image
    img_read = skimage.io.imread(filepath + image)
    img_double = double_img(img_read)
    source_img = orientation_standard(img_double)

    # show masked image
    fig = plt.figure()
    dispImage(source_img)
    brain_contour = plt.contour(mask_brain, colors='black')
    spine_contour = plt.contour(mask_spine, colors='black')
    
tes_img = image_import_overlay(atlas_data_str, 'atlas_1_BRAIN_STEM.png', 'atlas_1_SPINAL_CORD.png', 'atlas_1.png')

#%%
# display CT images with the spinal cord and brain stem images overlaid as contours 
# each patient in a seperate figure

# atlas data
atlas_1 = image_import_overlay(atlas_data_str, 'atlas_1_BRAIN_STEM.png', 'atlas_1_SPINAL_CORD.png', 'atlas_1.png')
atlas_2 = image_import_overlay(atlas_data_str, 'atlas_2_BRAIN_STEM.png', 'atlas_2_SPINAL_CORD.png', 'atlas_2.png')
atlas_3 = image_import_overlay(atlas_data_str, 'atlas_3_BRAIN_STEM.png', 'atlas_3_SPINAL_CORD.png', 'atlas_3.png')
atlas_4 = image_import_overlay(atlas_data_str, 'atlas_4_BRAIN_STEM.png', 'atlas_4_SPINAL_CORD.png', 'atlas_4.png')
atlas_5 = image_import_overlay(atlas_data_str, 'atlas_5_BRAIN_STEM.png', 'atlas_5_SPINAL_CORD.png', 'atlas_5.png')

# test data
test_1 = image_import_overlay(test_data_str, 'test_1_BRAIN_STEM.png', 'test_1_SPINAL_CORD.png', 'test_1.png')
test_2 = image_import_overlay(test_data_str, 'test_2_BRAIN_STEM.png', 'test_2_SPINAL_CORD.png', 'test_2.png')
test_3 = image_import_overlay(test_data_str, 'test_3_BRAIN_STEM.png', 'test_3_SPINAL_CORD.png', 'test_3.png')
test_4 = image_import_overlay(test_data_str, 'test_4_BRAIN_STEM.png', 'test_4_SPINAL_CORD.png', 'test_4.png')
test_5 = image_import_overlay(test_data_str, 'test_5_BRAIN_STEM.png', 'test_5_SPINAL_CORD.png', 'test_5.png')

# tune data
tune_1 = image_import_overlay(tune_data_str, 'tune_1_BRAIN_STEM.png', 'tune_1_SPINAL_CORD.png', 'tune_1.png')
tune_2 = image_import_overlay(tune_data_str, 'tune_2_BRAIN_STEM.png', 'tune_2_SPINAL_CORD.png', 'tune_2.png')
tune_3 = image_import_overlay(tune_data_str, 'tune_3_BRAIN_STEM.png', 'tune_3_SPINAL_CORD.png', 'tune_3.png')

# make lists
atlas_list = [atlas_1, atlas_2, atlas_3, atlas_4, atlas_5]
test_list = [test_1, test_2, test_3, test_4, test_5]
tune_list = [tune_1, tune_2, tune_3]


# %%
from demonsReg import demonsReg

n = 0
while n<len(tune_list):
    while n<len(atlas_list):
        demonsReg(tune_list[n], atlas_list[n])
        plt.close('all')
        n = n+1
    n = n+1

        

# %%
