"""
function to peform a registration between two 2D images using the demons  algorithm
 
provided for use in image registration exercises 3 for module MPHY0025 (IPMI)

Jamie McClelland
UCL
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize, rescale
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interpn
from utilsCoursework import dispImage, resampImageWithDefField, calcMSD, dispDefField

def demonsReg(source, target, sigma_elastic=1, sigma_fluid=1, num_lev=3, use_composition=False,
              use_target_grad=False, max_it=1000, check_MSD=True, disp_freq=5, disp_spacing=2, 
              scale_update_for_display=10, disp_method_df='grid', disp_method_up='arrows'):
  """
  Perform a registration between the 2D source image and the 2D target
  image using the demons algorithm. The source image is warped (resampled)
  into the space of the target image.
  
  The final warped image and deformation field can be returned as outputs
  from the function.
    
  The values of sigma_elastic and sigma_fluid can optionally be provided
  to specify the amount of elastic and fluid regularistion to apply. these
  values specify the standard deviation of the Gaussian used to smooth the
  update (fluid) or displacement field (elastic). a value of 0 means no
  smoothing is applied. default values are:
  sigma_elastic = 1
  sigma_fluid = 1
    The registration uses a multi-resolution scheme. The num_lev parameter
  can be used to specify the number of resolution levels to use. The
  default number of levels to use is 3.
   The demons registration can be performed by either adding (classical
  demons) or composing (diffeomorphic demons) the updates at each
  iteration. Set the use_composition parameter to true to compose the
  updates at each iteration. The default value for use_composition is
  false, i.e. the updates will be added by default.
    
   There are a number of other parameters affecting the registration or
  how the results are displayed, which are explained below. These can be
  speficied using name-value pair arguments, e.g.:
  demonsReg(..., 'max_it', 500)
  The default values are given after the parameter name
    use_target_grad = false
        logical (true/false) value indicating whether the target image
        gradient or source image gradient is used when calculating the
        demons forces.
    max_it = 1000
        the maximum number of iterations to perform.
    check_MSD = true
        logical value indicating if the Mean Squared Difference (MSD)
        should be checked for improvement at each iteration. If true, the
        MSD will be evaluated at each iteration, and if there is no
        improvement since the previous iteration the registration will move
        to the next resolution level or finish if it is on the final level.
    disp_freq = 5
        the frequency with which to update the displayed images. the images
        will be updated every disp_freq iterations. If disp_freq is set to
        0 the images will not be displayed during the registration
    disp_spacing = 2
        the spacing between the grid lines or arrows when displaying the
        deformation field and update.
    scale_update_for_display = 10
        the factor used to scale the update field for displaying
    disp_method_df = 'grid'
        the display method for the deformation field.
        can be 'grid' or 'arrows'
    disp_method_up = 'arrows'
        the display method for the update. can be 'grid' or 'arrows'
  """
  
  # make copies of full resolution images
  source_full = source;
  target_full = target;
  
  # loop over resolution levels
  for lev in range(1, num_lev + 1):
    
    # resample images if not final level
    if lev != num_lev:
      resamp_factor = np.power(2, num_lev - lev)
      target = rescale(target_full, 1.0 / resamp_factor, mode='edge', order=3, anti_aliasing=True)
      source = rescale(source_full, 1.0 / resamp_factor, mode='edge', order=3, anti_aliasing=True)
    else:
      target = target_full
      source = source_full
      
    # if first level initialise def_field and disp_field
    if lev == 1:
      [X, Y] = np.mgrid[0:target.shape[0], 0:target.shape[1]]
      def_field = np.zeros((X.shape[0], X.shape[1], 2))
      def_field[:, :, 0] = X
      def_field[:, :, 1] = Y
      disp_field_x = np.zeros(target.shape)
      disp_field_y = np.zeros(target.shape)
    else:
      # otherwise upsample disp_field from previous level
      disp_field_x = 2 * resize(disp_field_x, target.shape, mode='edge', order=3)
      disp_field_y = 2 * resize(disp_field_y, target.shape, mode='edge', order=3)
      # recalculate def_field for this level from disp_field
      X, Y = np.mgrid[0:target.shape[0], 0:target.shape[1]]
      def_field = np.zeros((X.shape[0], X.shape[1], 2))  # clear def_field from previous level
      def_field[:, :, 0] = X + disp_field_x
      def_field[:, :, 1] = Y + disp_field_y
    
    #initialise updates
    update_x = np.zeros(target.shape)
    update_y = np.zeros(target.shape)
    update_def_field = np.zeros(def_field.shape)
    
    # calculate the transformed image at the start of this level
    warped_image = resampImageWithDefField(source, def_field)
    
    # store the current def_field and MSD value to check for improvements at 
    # end of iteration 
    def_field_prev = def_field.copy()
    prev_MSD = calcMSD(target, warped_image)
        
    # pre-calculate the image gradients. only one of source or target
    # gradients needs to be calculated, as indicated by use_target_grad
    if use_target_grad:
      [target_gradient_x, target_gradient_y] = np.gradient(target)
    else:
      [source_gradient_x, source_gradient_y] = np.gradient(source)
        
    if disp_freq > 0:
      # DISPLAY RESULTS
      # figure 1 - source image (does not change during registration)
      # figure 2 - target image (does not change during registration)
      # figure 3 - source image transformed by current deformation field
      # figure 4 - deformation field
      # figure 5 - update
      plt.figure(1)
      plt.clf()
      dispImage(source)
      plt.pause(0.05)
      plt.figure(2)
      plt.clf()
      dispImage(target)
      plt.pause(0.05)
      plt.figure(3)
      plt.clf()
      dispImage(warped_image)
      x_lims = plt.xlim()
      y_lims = plt.ylim()
      plt.pause(0.05)
      plt.figure(4)
      plt.clf()
      dispDefField(def_field, spacing=disp_spacing, plot_type=disp_method_df)
      plt.xlim(x_lims)
      plt.ylim(y_lims)
      plt.pause(0.05)
      plt.figure(5)
      plt.clf()
      up_field_to_display = scale_update_for_display * np.dstack((update_x, update_y))
      up_field_to_display += np.dstack((X, Y))
      dispDefField(up_field_to_display, spacing=disp_spacing, plot_type=disp_method_up)
      plt.xlim(x_lims)
      plt.ylim(y_lims)
      plt.pause(0.05)
      
      # if first level pause so user can position figure
      if lev == 1:
        input('position the figures as desired and then push enter to run the registration')
      
    # main iterative loop - repeat until max number of iterations reached
    for it in range(max_it):
      
      # calculate update from demons forces
      #
      # if using target image graident use as is
      if use_target_grad:
        img_grad_x = target_gradient_x
        img_grad_y = target_gradient_y
      else:
        # but if using source image gradient need to transform with
        # current deformation field
        img_grad_x = resampImageWithDefField(source_gradient_x, def_field)
        img_grad_y = resampImageWithDefField(source_gradient_y, def_field)
        
      # calculate difference image
      diff = target - warped_image
      # calculate denominator of demons forces
      denom = np.power(img_grad_x, 2) + np.power(img_grad_y, 2) + np.power(diff, 2)
      # calculate x and y components of numerator of demons forces
      numer_x = diff * img_grad_x
      numer_y = diff * img_grad_y
      # calculate the x and y components of the update
      update_x = numer_x / denom
      update_y = numer_y / denom
      # set nan values to 0
      update_x[np.isnan(update_x)] = 0
      update_y[np.isnan(update_y)] = 0
            
      # if fluid like regularisation used smooth the update
      if sigma_fluid > 0:
        update_x = gaussian_filter(update_x, sigma_fluid, mode='nearest')
        update_y = gaussian_filter(update_y, sigma_fluid, mode='nearest')
      
      # update displacement field using addition (original demons) or
      # composition (diffeomorphic demons)
      if use_composition:
        # compose update with current transformation - this is done by
        # transforming (resampling) the current transformation using the
        # update. we can use the same function as used for resampling
        # images, and treat each component of the current deformation
        # field as an image
        # the update is a displacement field, but to resample an image
        # we need a deformation field, so need to calculate deformation
        # field corresponding to update.
        update_def_field[:, :, 0] = update_x + X
        update_def_field[:, :, 1] = update_y + Y
        # use this to resample the current deformation field, storing
        # the result in the same variable, i.e. we overwrite/update the
        # current deformation field with the composed transformation
        # scipy interpn function rather than resampImageWithDefField so that
        # values outside the def field are calculated via extrapolation rather
        # than being set to NaN
        x_coords = np.arange(def_field.shape[0], dtype = 'float')
        y_coords = np.arange(def_field.shape[1], dtype = 'float')
        def_field[:, :, 0] = interpn((x_coords, y_coords), def_field[:, :, 0], update_def_field, fill_value=None, bounds_error=False, method='linear')
        def_field[:, :, 1] = interpn((x_coords, y_coords), def_field[:, :, 1], update_def_field, fill_value=None, bounds_error=False, method='linear')
        # calculate the displacement field from the composed deformation field
        disp_field_x = def_field[:, :, 0] - X
        disp_field_y = def_field[:, :, 1] - Y
      else:
        # add the update to the current displacement field
        disp_field_x = disp_field_x + update_x
        disp_field_y = disp_field_y + update_y
      
      
      # if elastic like regularisation used smooth the displacement field
      if sigma_elastic > 0:
        disp_field_x = gaussian_filter(disp_field_x, sigma_elastic, mode='nearest')
        disp_field_y = gaussian_filter(disp_field_y, sigma_elastic, mode='nearest')
      
      # update deformation field from disp field
      def_field[:, :, 0] = disp_field_x + X
      def_field[:, :, 1] = disp_field_y + Y
            
      # transform the image using the updated deformation field
      warped_image = resampImageWithDefField(source, def_field)

      # update images if required for this iteration
      if disp_freq > 0 and it % disp_freq == 0:
        plt.figure(3)
        dispImage(warped_image)
        plt.pause(0.05)
        plt.figure(4)
        plt.clf()
        dispDefField(def_field, spacing=disp_spacing, plot_type=disp_method_df)
        plt.xlim(x_lims)
        plt.ylim(y_lims)
        plt.pause(0.05)
        plt.figure(5)
        plt.clf()
        up_field_to_display = scale_update_for_display * np.dstack((update_x, update_y))
        up_field_to_display += np.dstack((X, Y))
        dispDefField(up_field_to_display, spacing=disp_spacing, plot_type=disp_method_up)
        plt.xlim(x_lims)
        plt.ylim(y_lims)
        plt.pause(0.05)
      
      # calculate MSD between target and warped image
      MSD = calcMSD(target, warped_image)

      # display numerical results
      print('Level {0:d}, Iteration {1:d}: MSD = {2:f}\n'.format(lev, it, MSD))
      
      # check for improvement in MSD if required
      if check_MSD and MSD >= prev_MSD:
        # restore previous results and finish level
        def_field = def_field_prev
        warped_image = resampImageWithDefField(source, def_field)
        print('No improvement in MSD')
        break
      
      # update previous values of def_field and MSD
      def_field_prev = def_field.copy()
      prev_MSD = MSD.copy()
      
  # display the final results
  if disp_freq > 0:
    plt.figure(3)
    dispImage(warped_image)
    plt.figure(4)
    plt.clf()
    dispDefField(def_field, spacing=disp_spacing, plot_type=disp_method_df)
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.figure(5)
    plt.clf()
    up_field_to_display = scale_update_for_display * np.dstack((update_x, update_y))
    up_field_to_display += np.dstack((X, Y))
    dispDefField(up_field_to_display, spacing=disp_spacing, plot_type=disp_method_up)
    plt.xlim(x_lims)
    plt.ylim(y_lims)

  # return the transformed image and the deformation field
  return warped_image, def_field

