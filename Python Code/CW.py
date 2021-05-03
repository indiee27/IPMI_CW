#%%
%matplotlib qt

#%%
#imports
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
    transpose_image = np.transpose(file1)
    # flip along second dimension - top to bottom
    flipped_image = np.flip(transpose_image, 1)
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
#imports
from demonsReg import demonsReg
import numpy.ma as ma 
from utilsCoursework import resampImageWithDefField

#%%
# loop for default parameters
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
        img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0)

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


#%%
#iteration lists

num_lev_list = [1, 2, 3, 4, 5]
sigma_fluid_list = [0, 0.5, 1]
sigma_elastic_list = [0, 0.5, 1]

#%%

# loop to test parameters
for x in num_lev_list:
    for y in sigma_fluid_list:
        for z in sigma_elastic_list:
            
            # use_comp = True 
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
                    img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, num_lev = x, use_composition=True, sigma_fluid = y, sigma_elastic = z)

                    #names
                    warped_name = 'lev' + str(x) + 'elastic' + str(y) + 'fluid' + str(z) + 'usecomp_True_' + tune_name + atlas_name + 'warped.png'
                    result_name = 'lev' + str(x) + 'elastic' + str(y) + 'fluid' + str(z) + 'usecomp_True_' + tune_name + atlas_name + 'result.png'

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

            # use_comp = False
            a = 0
            while a<len(tune_list):
                #display the overlaid tuning image
                tune_name = str(tune_list[a])
                tune_file_loc = 'overlaid_images/' + tune_name
                t_o_name = tune_file_loc + 'overlaid.png'
                tune_overlay = skimage.io.imread(t_o_name, as_gray=True)
                #dispImage(tune_overlay)

                #import the source  
                t_s_name = tune_file_loc + 'source.png'
                tune_source = skimage.io.imread(t_s_name, as_gray=True)

                b = 0
                while m<len(atlas_list):
                    # display overlaid atlas image
                    atlas_name = str(atlas_list[b])
                    atlas_file_loc = 'overlaid_images/' + atlas_name
                    a_s_name = atlas_file_loc + 'source.png'
                    atlas_source = skimage.io.imread(a_s_name, as_gray=True)

                    a_m_name = atlas_file_loc + 'mask.png'
                    atlas_mask = skimage.io.imread(a_m_name, as_gray=True)
                    atlas_mask[atlas_mask<1] = 0
                    atlas_mask = np.flip(atlas_mask,0)

                    # demons reg
                    img_warped, img_def = demonsReg(atlas_source, tune_source, disp_freq=0, num_lev = x, use_composition=False, sigma_fluid = y, sigma_elastic = z)

                    #names
                    warped_name = 'lev' + str(x) + 'elastic' + str(y) + 'fluid' + str(z) + 'usecomp_False_' + tune_name + atlas_name + 'warped.png'
                    result_name = 'lev' + str(x) + 'elastic' + str(y) + 'fluid' + str(z) + 'usecomp_False_' + tune_name + atlas_name + 'result.png'

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
                    a = a + 1
                b = b + 1

# %%
#########################################
# SECTION 1.3 
#########################################
# %%
#imports
from PIL import Image
from utilsCoursework import calcLMSD

#%%
# write binary check function
def is_binary(file):
    img = file
    w,h = img.shape
    for i in range(w):
        for j in range(h):
            x = (i,j)
            if x == 1 or x == 0:
                pass
            else:
                IOError            

#%%
# write multi atlas segmentation function
def multi_atlas_segment(atlas_list, target_name):
    # import target ct image
    target_file_loc = 'overlaid_images/' + target_name
    t_s_name = target_file_loc + 'source.png'
    target_source = skimage.io.imread(t_s_name, as_gray=True)
    
    # initialise lists
    msd_results = []
    reg_weight_results = []
    brain_warp_results = []
    spine_warp_results = []

    # loop through atlas patients
    n=0 
    while n < len(atlas_list):
        # import atlas ct image
        atlas_name = str(atlas_list[n])
        atlas_file_loc = 'overlaid_images/' + atlas_name
        a_s_name = atlas_file_loc + 'source.png'
        atlas_source = skimage.io.imread(a_s_name, as_gray=True)
       
        # registration
        img_warped, img_def = demonsReg(atlas_source, target_source, disp_freq=0, num_lev=5, sigma_elastic=1, sigma_fluid=1, use_composition=True)

        # import spinal cord and brain stem binary images
        brain = skimage.io.imread('masks/' + atlas_name + 'brain.png', as_gray=True)
        spine = skimage.io.imread('masks/' + atlas_name + 'spine.png', as_gray=True)

        # make sure the edges are binary values
        brain[brain>0.5] = 1
        brain[brain<0.5] = 0
        spine[spine>0.5] = 1
        spine[spine<0.5] = 0

        # warp binary images
        brain_warp = resampImageWithDefField(brain, img_def)
        spine_warp = resampImageWithDefField(spine, img_def)

        # check warps are binary
        is_binary(brain_warp)
        is_binary(spine_warp)

        #calculate lmsd between target and warped
        msd = calcLMSD(target_source, img_warped, 20)

        #reg weight calculation
        w,h = msd.shape
        registration_weight = np.ndarray([w,h])
        for i in range(w):
            for j in range(h):
                x = msd[i][j]
                x_reg = 1/(1+x)
                registration_weight[i][j] = x_reg
        
        #save msd and reg arrays from the atlas loop
        msd_results.append(msd)
        reg_weight_results.append(registration_weight)
        brain_warp_results.append(brain_warp)
        spine_warp_results.append(spine_warp)

        # next atlas
        n = n + 1
    
    # sum current weights over all registrations
    w,h = target_source.shape
    sum_registration_weights = np.ndarray([w,h])
    for a in range(len(atlas_list)):
        sum_registration_weights = np.sum(reg_weight_results[a])

    # normalise registration weights
    w,h = target_source.shape
    a = len(atlas_list)
    reg_weight_norm = []
    for a in range(len(atlas_list)):
        atlas_item = reg_weight_results[a]
        reg_weight_norm_item = np.ndarray([w,h])
        for i in range(w):
            for j in range(h):
                x = atlas_item[i][j]
                y = sum_registration_weights[i][j]
                value = x/y
                reg_weight_norm_item[i][j] = value 
        reg_weight_norm.append(reg_weight_norm_item)
    
    # mas prob calculation brain
    w,h = target_source.shape
    MAS_brain = np.ndarray([w,h])
    a = len(atlas_list)
    mas_brain_list = []
    #multiply values for each atlas
    for a in range(a):
        reg_weight_item = reg_weight_norm[a]
        brain_warp_item = brain_warp_results[a]
        mas_brain_item = np.ndarray([w,h])
        for i in range(w):
            for j in range(h):
                x = reg_weight_item[i][j]
                y = brain_warp_item[i][j]
                value = x * y
                mas_brain_item[i][j] = value
        mas_brain_list.append(mas_brain_item)
    #sum atlas values
    for a in range(a):
        MAS_brain[i][j] = np.sum(mas_brain_list[a])      

    # mas prob calculation spine
    w,h = target_source.shape
    MAS_spine = np.ndarray([w,h])
    a = len(atlas_list)
    mas_spine_list = []
    #multiply values for each atlas
    for a in range(a):
        reg_weight_item = reg_weight_norm[a]
        spine_warp_item = spine_warp_results[a]
        mas_spine_item = np.ndarray([w,h])
        for i in range(w):
            for j in range(h):
                x = reg_weight_item[i][j]
                y = spine_warp_item[i][j]
                value = x * y
                mas_spine_item[i][j] = value
        mas_spine_list.append(mas_spine_item)
    #sum atlas values
    for a in range(a):
        MAS_spine[i][j] = np.sum(mas_spine_list[a])
    
    #generate binary segmentation images
    # brain stem
    MAS_brain[MAS_brain>=0.5] = 1
    MAS_brain[MAS_brain<0.5] = 0
    MAS_spine[MAS_spine>=0.5] = 1
    MAS_spine[MAS_spine<0.5] = 0
    
    return MAS_brain, MAS_spine, target_source 

#%%

tuning_1 = 'tune_1_'
brain, spine, ct_scan = multi_atlas_segment(atlas_list, tuning_1)
#%%
brain_og = skimage.io.imread('masks/tune_1_brain.png', as_gray=True)
spine_og = skimage.io.imread('masks/tune_1_spine.png', as_gray=True)
plt.figure()
axim, ax = plt.subplots()
ax.imshow(ct_scan, cmap='gray')
ax.contour(brain_og, linewidths=1, colors=['green'])
ax.contour(spine_og, linewidths=1, colors=['red'])
ax.contour(brain, linewidths=1, colors=['magenta'])
ax.contour(spine, linewidths=1, colors=['cyan'])               
# %%
