"""Script that implements the calibration process required for lab 8."""

import cv2 as cv


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.

# Need at least 10 test patterns for camera calibration. NOTE: Thought we only
# needed 6 (as each point correspondence gives us 2 linearly independent
# equations), but there are more unknowns we're solving for here (distortion).

# Most importantly, need 3D points and the 2D points in the image they
# correspond to.

