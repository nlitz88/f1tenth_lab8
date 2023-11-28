"""Script that implements the calibration process required for lab 8."""

from typing import Tuple
import cv2 as cv
import numpy as np
from pathlib import Path


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.



# def get_chessboard_image_points(image: np.ndarray,
#                                 pattern_size: Tuple(),


#                       )
# retval, corners = cv.findChessboardCorners(image=None,
#                                            patternSize=(),
#                                            corners=None,
#                                            flags=None)

# 2. Solve for camera intrinsic matrix (K).
# Using the 3D-->2D point correspondences obtained across all the images in the
# previous step.


# def world_points_from_chessboard_corners(chessboard_corners: np.ndarray) -> list:

#     # Essentially, we already know exactly what this array is going to look
#     # like, whether we know the square size or not. Why? Because in every image,
#     # we're expecting to find corner points for the exact same pattern--a
#     # pattern that we already know.

#     # Therefore, if the pattern is 6x8 (for example), then we know that we're
#     # ALWAYS going to get 48 corner points if the pattern was found.

#     # AND, we know the first row of those corner points is separated by 1
#     # square, we know each row is separated by one square, and we know the top
#     # leftmost point is the world origin (0,0,0). Therefore, because we know
#     # what the 3D points are going to be right off the bat, we can just add the
#     # same array of 3D points for every image, as long as the pattern is found!

#     # The reason these same 3D points are still useful (even though they're the
#     # same across all our images) is because they map back to different image
#     # points! (therefore giving us linearly independent equations from different
#     # point correspondences). 


#     # Okay, they generate this using a meshgrid--I guess I could see how that
#     # works.
    

#     pass


if __name__ == "__main__":

    # TODO: Add CLI here for parameterizing this calibration script.

    # Set path to calibration image directory.
    calibration_image_dir = Path(r"./calibration")
    calibration_image_paths = [path for path in calibration_image_dir.iterdir() if path.is_file()]
    
    # Define other calibration parameters.
    chessboard_internal_corners_row_width = 6
    chessboard_internal_corners_column_height = 8
    chessboard_size = (chessboard_internal_corners_row_width, chessboard_internal_corners_column_height)
    # Also copied from tutorial: Corner point refining steps.
    corner_point_refining_window_size = 11
    corner_point_refining_criteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    debug_display_corners = True

    # Create lists for resulting 3D points and 2D points.
    world_points = []
    image_points = []

    # Precompute the 3D point array corresponding to each of the hxw corner
    # points found in a single image. The tutorial constructs it rather
    # compactly, so using their code here:
    chessboard_world_points = np.zeros((chessboard_internal_corners_row_width*chessboard_internal_corners_column_height, 3), np.float32)
    chessboard_world_points[:,:2] = np.mgrid[0:chessboard_internal_corners_column_height,0:chessboard_internal_corners_row_width].T.reshape(-1,2)

    # For each image:
    for image_path in calibration_image_paths:
        
        # 1. Load the current image into memory.
        image = cv.imread(str(image_path))
        # cv.imshow(image_path.parts[-1], image)
        # cv.waitKey(1000)

        # 2. Attempt to find the corners points in the chessboard image.
        retval, corners = cv.findChessboardCorners(image=image,
                                                   patternSize=chessboard_size)
        
        # 3. If findChessboardCorners was able to locate the corners within the
        #    chessboard image, map those corner points to their 3D world points
        #    and store them away alongside their respective 2D image points.
        if retval == True:

            # Map those corner points to 3D world points based on their relative
            # position to the top-left-most corner point. Because these are all
            # measured from the origin of the world frame, these will be the
            # same for every image that corners are found in. Therefore, just
            # take the 3D point array computed above (that is common to every
            # chessboard image with corners) and add it to the list of 3D world
            # points.
            world_points.append(chessboard_world_points)

            # Refine the 2D image points using cornerSubPix according to
            # tutorial.
            # Create a grayscale version of the image to use with cornerSubPix
            # function.
            image_gray = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY)
            refined_image_corner_points = cv.cornerSubPix(image=image_gray, 
                                                          corners=corners,
                                                          winSize=(corner_point_refining_window_size,corner_point_refining_window_size),
                                                          zeroZone=(-1,-1),
                                                          criteria=corner_point_refining_criteria)

            # Add the newly refined 2D corner point to the list of image points.
            image_points.append(refined_image_corner_points)

            # Display corners if enabled.
            if debug_display_corners == True:
                # Draw the refined corners on the the original color image.
                cv.drawChessboardCorners(image=image,
                                         patternSize=chessboard_size,
                                         corners=refined_image_corner_points,
                                         patternWasFound=True)# 
                # Display the updated (drawn on) image.
                cv.imshow(image_path.parts[-1], image)
                cv.waitKey(500)

    # 1. Find corner points in each image.
    # First, need to find the corners of the chessboard in each of the provided
    # images. Can use the opencv find chessboard corners function.
    # image_corners = 


    # The opencv function we use assumes the top left-most corner is the
    # world coordinate origin, and assumes that the chessboard lies flat on the XY
    # plane of the world frame such that all points on the chessboard have Zw == 0.
    # Based on that assumption, once we detect all the points on our chessboard from
    # left to right, top to bottom--for each of those points, we can compute its
    # position relative to the origin (top-left) point.
    
    
    cv.destroyAllWindows()