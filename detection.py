"""Functions for performing inference on TensorRT model. In particular, running
inference on our F1Tenth Car Detection (Object Detection) model.
"""

# 1. Create a function to load the TensorRT model.

# 2. Create a function where we invoke the TensorRT "engine" (model). Will
#    likely have to use the TensorRT python package.

# 3. Create "preprocessing" function. I.e., can resize images if need be. Can
#    use torch transformations, maybe opencv. Depends on what we need.

# 4. Create a "postprocessing" function where we draw the predicted bounding box
#    onto the image (can use openCV to do this).