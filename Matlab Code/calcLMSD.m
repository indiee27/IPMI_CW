function lmsd_map = calcLMSD(image1, image2, win_sz)
%function to calculate the local-mean-squared-differences between two
%images at every pixel using a box window
%
%INPUTS:    image1: the first image
%           image2: the second image
%           win_sz: the size of the window used for calculating the LNCC
%OUTPUTS:   lmsd_map: the value of the LMSD at each pixel
%
%NOTES: for each pixel the LMSD is calculated as the MSD between two
%sub-images, that extend win_sz to the left/right/above/below the pixel, so
%the total size of the sub-images is 2*win_sz + 1. If there are less than
%win_sz pixels on one side, then the sub-image will be truncated to the
%number of pixels that are available.


%loop over all pixels in images
im_size = size(image1);
lmsd_map = nan(im_size);
for x = 1:im_size(1)
    for y = 1:im_size(2)
        
        %find first and last pixel to use in x and y direction
        first_x = x - win_sz;
        if first_x < 1
            first_x = 1;
        end
        last_x = x + win_sz;
        if last_x > im_size(1)
            last_x = im_size(1);
        end
        first_y = y - win_sz;
        if first_y < 1
            first_y = 1;
        end
        last_y = y + win_sz;
        if last_y > im_size(2)
            last_y = im_size(2);
        end
        
        %form sub-images
        im1_win = image1(first_x:last_x, first_y:last_y);
        im2_win = image2(first_x:last_x, first_y:last_y);
        
        %calculate msd between subimages and save as lmsd for this pixel
        lmsd_map(x,y) = calcMSD(im1_win, im2_win);
    end
end

