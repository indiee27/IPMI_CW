#%%
%matplotlib qt

#%%
import numpy as np 
import matplotlib.pyplot as plt 
from utilsCoursework import dispImage
import skimage.io

#%%
#write file import function
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
# write overlay function
def import_overlay(filepath, brain, spine, image):
    '''
    function to take a CT scan and overlay contours of the brain and spine
    PARAMS:
    INPUT:  filepath:   string of file location
            brain:      image of brain mask 
            spine:      image of spine mask
            image:      CT scan to be overlaid
    OUTPUT: overlay_img:image of CT scan with overlaid contours
            source_img: imported orientated CT scan
            binary_img: binary image of combined contours
    '''
    #source
    source_img = import_double_orientate(filepath + image + '.png')
    name_source = str(image) + '_source' + '.png'
    img,plot = plt.subplots()
    plot.set_axis_off()
    img.add_axes(plot)
    plot.imshow(source_img, cmap='gray')
    plt.savefig(name_source, bbox_inches='tight', pad_inches=0)

    #mask
    brain_mask = import_double_orientate(filepath + brain)
    mi, ma = np.floor(np.nanmin(brain_mask)), np.ceil(np.nanmax(brain_mask))
    levels = np.arange(mi, ma+2, 2)

    spine_mask = import_double_orientate(filepath + spine)
    mi2, ma2 = np.floor(np.nanmin(spine_mask)), np.ceil(np.nanmax(spine_mask))
    levels2 = np.arange(mi2, ma2+2, 2)

    total_mask = spine_mask + brain_mask

    plots,ax = plt.subplots()
    ax.set_axis_off()
    plots.add_axes(ax)
    ax.contour(total_mask, linewidths=1, colors=['black'])
    binary_name = str(image) + '_mask' + '.png'
    plt.savefig(binary_name, bbox_inches='tight', pad_inches=0)
    binary_img = np.asarray(plots)

    #overlay
    ax_im, ax = plt.subplots()
    ax.set_axis_off()
    ax_im.add_axes(ax)
    ax.imshow(source_img, cmap='gray')
    ax.contour(brain_mask, levels=levels, linewidths=1, colors=['black'])
    ax.contour(spine_mask, levels=levels2, linewidths=1, colors=['black'])
    name_overlay = str(image) + '_overlaid' + '.png'
    plt.savefig(name_overlay, bbox_inches='tight', pad_inches=0)
    overlay_img = np.asarray(ax_im)

    return source_img, binary_img, overlay_img


#%%
# test data
test_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/test/")

test_1 = import_overlay(test_data_str, 'test_1_BRAIN_STEM.png', 'test_1_SPINAL_CORD.png', 'test_1')
test_2 = import_overlay(test_data_str, 'test_2_BRAIN_STEM.png', 'test_2_SPINAL_CORD.png', 'test_2')
test_3 = import_overlay(test_data_str, 'test_3_BRAIN_STEM.png', 'test_3_SPINAL_CORD.png', 'test_3')
test_4 = import_overlay(test_data_str, 'test_4_BRAIN_STEM.png', 'test_4_SPINAL_CORD.png', 'test_4')
test_5 = import_overlay(test_data_str, 'test_5_BRAIN_STEM.png', 'test_5_SPINAL_CORD.png', 'test_5')

# %%
# atlas data
atlas_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/atlas/")

atlas_1 = import_overlay(atlas_data_str, 'atlas_1_BRAIN_STEM.png', 'atlas_1_SPINAL_CORD.png', 'atlas_1')
atlas_2 = import_overlay(atlas_data_str, 'atlas_2_BRAIN_STEM.png', 'atlas_2_SPINAL_CORD.png', 'atlas_2')
atlas_3 = import_overlay(atlas_data_str, 'atlas_3_BRAIN_STEM.png', 'atlas_3_SPINAL_CORD.png', 'atlas_3')
atlas_4 = import_overlay(atlas_data_str, 'atlas_4_BRAIN_STEM.png', 'atlas_4_SPINAL_CORD.png', 'atlas_4')
atlas_5 = import_overlay(atlas_data_str, 'atlas_5_BRAIN_STEM.png', 'atlas_5_SPINAL_CORD.png', 'atlas_5')

#%%
# tuning data
tune_data_str = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/tune/")

tune_1 = import_overlay(tune_data_str, 'tune_1_BRAIN_STEM.png', 'tune_1_SPINAL_CORD.png', 'tune_1')
tune_2 = import_overlay(tune_data_str, 'tune_2_BRAIN_STEM.png', 'tune_2_SPINAL_CORD.png', 'tune_2')
tune_3 = import_overlay(tune_data_str, 'tune_3_BRAIN_STEM.png', 'tune_3_SPINAL_CORD.png', 'tune_3')

# %%
#####################################################
# SECTION 1.2
#####################################################

#%%
# make lists to iterate through
tune_list = ['tune_1_','tune_2_','tune_3_']
atlas_list = ['atlas_1_','atlas_2_','atlas_3_','atlas_4_','atlas_5_']
test_list = ['test_1_','test_2_','test_3_','test_4_','test_5_']

#%%
#write file import function
def import_gray(file):
    ''' 
    function to import, double, and reorientate IN GRAYSCALE
    PARAMS:
    INPUT:  file:   image
    OUTPUT: file:   image
    '''
    img = skimage.io.imread(file, as_gray=True)
    file1 = np.double(img)
    # transpose - switch x and y dimensions
    #transpose_image = np.transpose(file1)
    # flip along second dimension - top to bottom
    #flipped_image = np.flip(file1,0)
    return file1

#%%
# wait
import msvcrt as m
def wait():
    m.getch()

#%%
from demonsReg import demonsReg

n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    f_t_name = tune_list[n]
    tune_img = import_gray(f_t_name)
    dispImage(tune_img)

    while n<len(atlas_list):
        # display overlaid atlas image
        f_a_name = atlas_list[n]
        atlas_img = import_gray(f_a_name)
        dispImage(atlas_img)

        # demons reg
        img_warped, img_def = demonsReg(atlas_img, tune_img)

        plt.savefig([f_t_name,f_a_name])

        # pause and ask to continue
        wait

        # close open figures
        plt.close('all')

        # next loop
        n = n + 1
    n = n + 1

# %%
