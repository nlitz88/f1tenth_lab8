"""Script that implements the calibration process required for lab 8."""

import cv2 as cv


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.

# Need at least 10 test patterns for camera calibration. NOTE: Thought we only
# needed 6 (as each point correspondence gives us 2 linearly independent
# equations), but there are more unknowns we're solving for here (distortion).

# Most importantly, need 3D points and the 2D points in the image they
# correspond to.

# An "easy" way of obtaining 3D points and corresponding 2D points is to use
# some sort of fiducial (like a checkerboard), or some sort of object with known
# dimensions. In the case of the checkerboard we'll use, we know the size of
# each square and we know where each square is with respect to the top-leftmost
# corner of the checkerboard, where that top left corner is what we'll treat as
# the world origin. Therefore, with one picture of a checkerboard, we'll get
# multiple 3D points to work with.

# The fact that we know the size of the squares allows us to figure out where
# the points on the board are from us. Why? Because the size of the square in
# the pixel/image plane is going to be some scale factor bigger or smaller than
# the object--where that scale factor is proportional to the distance. However,
# I'm not actually sure we have all the information we need to know what the 3D
# positions are--how are we possibly getting point correspondences then?

# TODO: Okay, a question I guess I have then: Looking at this opencv tutorial on
# calibration, how are we actually obtaining 3D points? I mean, again, assuming
# you know how big the object really is, in theory, you could determine how far
# away it is. But I feel like you'd at least need the focal length and pixel
# scale, no? 
# ANSWER: Correct, you need the camera intrinsics to be able to do this
# properly.
# https://stackoverflow.com/questions/59937813/finding-depth-of-an-object-in-an-image-given-coordinates-of-object-in-image-fram

# Okay, so if that's the case, then how are we getting 3D-->2D point
# correspondences if we can't obtain 3D locations of checkerboard locations yet?
# Or is there some trick we can use?

