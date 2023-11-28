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


def world_points_from_chessboard_corners(chessboard_corners: np.ndarray) -> list:

    pass


if __name__ == "__main__":

    # TODO: Add CLI here for parameterizing this calibration script.

    # Set path to calibration image directory.
    calibration_image_dir = Path(r"./calibration")
    calibration_image_paths = [path for path in calibration_image_dir.iterdir() if path.is_file()]
    
    # Define other calibration parameters.
    chessboard_internal_corners_row_width = 6
    chessboard_internal_corners_column_height = 8
    chessboard_size = (chessboard_internal_corners_row_width, chessboard_internal_corners_column_height)

    # Create lists for resulting 3D points and 2D points.
    world_points = []
    image_points = []

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
            # position to the top-left-most corner point.
            print(corners.shape)

            # Refine the 2D image points using cornerSubPix according to
            # tutorial.

            # Add the newly refined 2D corner point to the list of image points.
            
        # 3. 

        # 4. Store each corner's newly determined 3D world point.

        # 5. Store the corner's corresponding 2D image point.


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