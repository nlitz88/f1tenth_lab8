"""Functions for performing inference on TensorRT model. In particular, running
inference on our F1Tenth Car Detection (Object Detection) model.
"""

from pathlib import Path
import time
from PIL import Image
import numpy as np


# https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
# Slight more relevant object detection tutorial with YoloV3.
# https://github.com/NVIDIA/TensorRT/blob/release/8.4/samples/python/yolov3_onnx/onnx_to_tensorrt.py


# 1. Create a function to load the TensorRT model.


# 2. Create a function where we invoke the TensorRT "engine" (model). Will
#    likely have to use the TensorRT python package.

# 3. Create "preprocessing" function. I.e., can resize images if need be. Can
#    use torch transformations, maybe opencv. Depends on what we need.
# def preprocess_image(image: np.ndarray) -> np.ndarray:

# 4. Create a "postprocessing" function where we draw the predicted bounding box
#    onto the image (can use openCV to do this).


def main():

    pass

if __name__ == "__main__":

    trt_model_path = Path(r"./f1yolo_fp32.trt")
    test_image_path = Path(r"./resource/test_car_x60cm.png")

    model_input_shape_chw = (3, 180, 320)
    # Take the model input shape and convert it to the format Pillow accepts
    # (Width, Height). We can tell Pillow to resize 
    resize_shape_wh = (model_input_shape_chw[2], model_input_shape_chw[1])
    test_image = Image.open(test_image_path)
    test_image = test_image.resize(resize_shape_wh, resample=Image.BICUBIC)
    test_image = np.array(test_image, dtype=np.float32, order="C")

    # Reshape the np array that used to be a PIL image into "CHW" (PyTorch shape
    # convention).
    test_image = np.transpose(test_image, [2, 0, 1])
    # CHW to NCHW format. I.e., adding another dimension in case you had a batch
    # size greater than 1.
    test_image = np.expand_dims(test_image, axis=0)
    # Convert the image to row major order in memory.
    test_image = np.array(test_image, dtype=np.float32, order="C")

    # 