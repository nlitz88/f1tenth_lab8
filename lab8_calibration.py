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


if __name__ == "__main__":

    # TODO: Add CLI here for parameterizing this calibration script.

    # Set path to calibration image directory.
    calibration_image_dir = Path(r"./calibration")
    calibration_image_paths = [path for path in calibration_image_dir.iterdir() if path.is_file()]

    # Create lists for resulting 3D points and 2D points.
    world_points = []
    image_points = []

    # For each image:
    for image_path in calibration_image_paths:

        # 1. Find the corners points in the chessboard image.

        # 2. Map those corner points to 3D world points based on their relative
        #    position to the top-left-most corner point.

        # 3. Store each corner's newly determined 3D world point.

        # 4. Store the corner's corresponding 2D image point.



        # 3. Add 

    # 1. Find corner points in each image.
    # First, need to find the corners of the chessboard in each of the provided
    # images. Can use the opencv find chessboard corners function.
    image_corners = 


    # The opencv function we use assumes the top left-most corner is the
    # world coordinate origin, and assumes that the chessboard lies flat on the XY
    # plane of the world frame such that all points on the chessboard have Zw == 0.
    # Based on that assumption, once we detect all the points on our chessboard from
    # left to right, top to bottom--for each of those points, we can compute its
    # position relative to the origin (top-left) point.
    