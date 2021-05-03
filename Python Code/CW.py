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
    #plt.savefig(name_source, bbox_inches='tight', pad_inches=0)

    #mask
    brain_mask = import_double_orientate(filepath + brain)
    brain_name = str(image) + '_brain.png'
    a,b = plt.subplots()
    b.set_axis_off()
    a.add_axes(b)
    b.imshow(brain_mask)
    plt.savefig(brain_name, bbox_inches='tight', pad_inches=0)
    mi, ma = np.floor(np.nanmin(brain_mask)), np.ceil(np.nanmax(brain_mask))
    levels = np.arange(mi, ma+2, 2)

    spine_mask = import_double_orientate(filepath + spine)
    spine_name = str(image) + '_spine.png'
    c,d = plt.subplots()
    d.set_axis_off()
    c.add_axes(d)
    d.imshow(spine_mask)
    plt.savefig(spine_name, bbox_inches='tight', pad_inches=0)
    mi2, ma2 = np.floor(np.nanmin(spine_mask)), np.ceil(np.nanmax(spine_mask))
    levels2 = np.arange(mi2, ma2+2, 2)

    total_mask = spine_mask + brain_mask

    plots,ax = plt.subplots()
    ax.set_axis_off()
    plots.add_axes(ax)
    ax.contour(total_mask, linewidths=1, colors=['black'])
    binary_name = str(image) + '_mask' + '.png'
    #plt.savefig(binary_name, bbox_inches='tight', pad_inches=0)
    binary_img = np.asarray(plots)

    #overlay
    ax_im, ax = plt.subplots()
    ax.set_axis_off()
    ax_im.add_axes(ax)
    ax.imshow(source_img, cmap='gray')
    ax.contour(brain_mask, levels=levels, linewidths=1, colors=['black'])
    ax.contour(spine_mask, levels=levels2, linewidths=1, colors=['black'])
    name_overlay = str(image) + '_overlaid' + '.png'
    #plt.savefig(name_overlay, bbox_inches='tight', pad_inches=0)
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
from demonsReg import demonsReg
import numpy.ma as ma 
from utilsCoursework import resampImageWithDefField

#%%
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    t_o_name = tune_name + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_name + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        a_s_name = atlas_name + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_name + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it=200)

        #names
        warped_name = tune_name + atlas_name + 'warped.png'
        result_name = tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

# %%
# parameter test

# num_levs, use_composition, sigma_elastic and sigma_fluid
#HOW CAN I PUT EACH SET OF RESULTS IN A DIFFERENT FOLDER???
#manually name them
#use composition is true or false
#num levs is a number, default is 3
#sigma elastic and sigma fluid between 0 and 1, default is 1
#%%
#PARAM 1
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 5, use_composition=False, sigma_fluid = 0.5, sigma_elastic = 0.5)

        #names
        warped_name = 'param_1/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_1/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#%%

#PARAM 2
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 5, use_composition=True, sigma_fluid = 0.5, sigma_elastic = 0.5)

        #names
        warped_name = 'param_2/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_2/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 3
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 3, use_composition=False, sigma_fluid = 0, sigma_elastic = 0)

        #names
        warped_name = 'param_3/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_3/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 4
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 3, use_composition=False, sigma_fluid = 0, sigma_elastic = 0)

        #names
        warped_name = 'param_4/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_4/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 5
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 5, use_composition=False, sigma_fluid = 1, sigma_elastic = 1)

        #names
        warped_name = 'param_5/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_5/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 6
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 5, use_composition=True, sigma_fluid = 1, sigma_elastic = 1)

        #names
        warped_name = 'param_6/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_6/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 7
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 1, use_composition=False, sigma_fluid = 1, sigma_elastic = 1)

        #names
        warped_name = 'param_7/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_7/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 8 
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 1, use_composition=True, sigma_fluid = 1, sigma_elastic = 1)

        #names
        warped_name = 'param_8/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_8/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 9
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 1, use_composition=False, sigma_fluid = 0.5, sigma_elastic = 0.5)

        #names
        warped_name = 'param_9/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_9/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

#PARAM 10
n = 0
while n<len(tune_list):
    #display the overlaid tuning image
    tune_name = str(tune_list[n])
    tune_file_loc = 'overlaid_images/' + tune_name
    t_o_name = tune_file_loc + 'overlaid.png'
    tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
    #dispImage(tune_overlay)

    #import the source  
    t_s_name = tune_file_loc + 'source.png'
    tune_source = skimage.io.imread(t_s_name, as_gray=True)

    m = 0
    while m<len(atlas_list):
        # display overlaid atlas image
        atlas_name = str(atlas_list[m])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)

        a_m_name = atlas_file_loc + 'mask.png'
        atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
        atlas_mask[atlas_mask<1] = 0
        atlas_mask = np.flip(atlas_mask,0)

        # demons reg
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, max_it = 300, num_lev = 1, use_composition=True, sigma_fluid = 0.5, sigma_elastic = 0.5)

        #names
        warped_name = 'param_10/' + tune_name + atlas_name + 'warped.png'
        result_name = 'param_10/' + tune_name + atlas_name + 'result.png'

        #save the figures
        ax,plot = plt.subplots()
        plot.set_axis_off()
        ax.add_axes(plot)
        plot.imshow(img_warped, cmap='gray')
        plt.savefig(warped_name, bbox_inches='tight', pad_inches=0)      

        #warp contours
        warped_atlas_mask = resampImageWithDefField(atlas_mask, img_def)

        # mask resulting image
        result = ma.masked_where(warped_atlas_mask == 0, img_warped)
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.savefig(result_name, bbox_inches='tight', pad_inches=0)

        # pause and ask to continue
        #input('Press enter to continue')

        # close open figures
        plt.close('all')

        # next loop
        m = m + 1
    n = n + 1

# %%
##############################
### SECTION 1.3 ######
##############################
# %%
from PIL import Image

def is_binary(file):
    img = Image.open(file)
    w,h = img.size
    for i in range(w):
        for j in range(h):
            x = img.getpixel((i,j))
            if x == 1 or x == 0:
                pass
            else:
                IOError

from utilsCoursework import calcLMSD                

#%%
def multi_atlas_segment(atlas_list, target_name):
    # import target ct image
    target_file_loc = 'overlaid_images/' + target_name
    t_s_name = target_file_loc + 'source.png'
    target_source = skimage.io.imread(t_s_name, as_gray=True)
    
    # loop through atlas patients
    n=0 
    for n < len(atlas_list):
        # import atlas ct image
        atlas_name = str(atlas_list[n])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)
       
        # registration
        img_warped, img_def = demonsReg(atlas_source, target_source)

        # import spinal cord and brain stem binary images
        filepath = ("C:/Users/indie/Documents/GitHub/IPMI_CW/Data (part 1)/head_and_neck_images/atlas")
        brain = import_double_orientate(filepath + atlas_name + 'BRAIN_STEM.png')
        spine = import_double_orientate(filepath + atlas_name + 'SPINAL_CORD.png')

        # numbered names
        brain_name = brain_warp_ + str([n])
        spine_name = spine_warp_ + str([n])

        # warp binary images
        brain_name = resampImageWithDefField(brain, img_def)
        spine_name = resampImageWithDefField(spine, img_def)

        # check warps are binary
        is_binary(brain_name)
        is_binary(spine_name)

        #calculate lmsd between target and warped
        msd = calcLMSD(target_source, img_warped, 20)

        #reg weight calculation
        w,h = msd.size
        for i in w:
            for j in h:
                