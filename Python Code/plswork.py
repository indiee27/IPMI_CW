#%%
%matplotlib qt 

import matplotlib.pyplot as plt 
import skimage.io

img = skimage.io.imread('C:/Users/indie/Documents/Github/IPMI_CW/Data (part 1)/head_and_neck_images/atlas/atlas_1.png')

from utilsCoursework import dispImage

plt.figure()
dispImage(img)
# %%
