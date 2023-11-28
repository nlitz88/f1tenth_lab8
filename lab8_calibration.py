"""Script that implements the calibration process required for lab 8."""

from typing import Tuple
import cv2 as cv
import numpy as np
from pathlib import Path


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.

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
    debug_display_corners = False

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


    # Once we have enough (10) 3D-->2D point correspondences from corners in our
    # chessboard calibration images, we can call the calibrateCamera function to
    # obtain the camera matrix == the camera's intrinsic parameters.
    # TODO Why is calibrateCamera expecting the dimensions in reverse order?
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objectPoints=world_points,
                                                                        imagePoints=image_points,
                                                                        imageSize=image_gray.shape[::-1],
                                                                        cameraMatrix=None,
                                                                        distCoeffs=None)
    

    
    cv.destroyAllWindows()