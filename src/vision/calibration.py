"""Calibration module.

This module contains a handful of functions to help you obtain your camera's K
matrix (camera matrix containing your camera's instrinsic parameters). 

At it's core, it's really just implementing the calibration process outlined in the
opencv documentation
(https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html), but with a
few small abstractions on top.
"""
import cv2 as cv

