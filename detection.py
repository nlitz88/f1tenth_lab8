"""Functions for performing inference on TensorRT model. In particular, running
inference on our F1Tenth Car Detection (Object Detection) model.
"""

from pathlib import Path
import time
from PIL import Image
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
# Slight more relevant object detection tutorial with YoloV3.
# https://github.com/NVIDIA/TensorRT/blob/release/8.4/samples/python/yolov3_onnx/onnx_to_tensorrt.py
#https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

# 1. Create a function to load the TensorRT model.

# 2. Create a function where we invoke the TensorRT "engine" (model). Will
#    likely have to use the TensorRT python package.

# 3. Create "preprocessing" function. I.e., can resize images if need be. Can
#    use torch transformations, maybe opencv. Depends on what we need.
# def preprocess_image(image: np.ndarray) -> np.ndarray:

# 4. Create a "postprocessing" function where we draw the predicted bounding box
#    onto the image (can use openCV to do this).

# Taken from
# https://github.com/NVIDIA/TensorRT/blob/release/8.4/samples/python/common.py#L123
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Taken from
# https://github.com/NVIDIA/TensorRT/blob/release/8.4/samples/python/common.py#L136
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.get_tensor_mode(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def main():

    trt_model_path = Path(r"./f1yolo_fp32.trt")
    test_image_path = Path(r"./resource/test_car_x60cm.png")
    # Expected output shape.
    # For our model, we only have a single detection head, and it has dimensions
    # b, 5, 6, 10. In this case, (1,5,6,10)
    output_shape = (1,5,6,10)
    # Model input shape (b, c, h, w)
    model_input_shape = (1, 3, 180, 320)
    # Take the model input shape and convert it to the format Pillow accepts
    # (Width, Height). We can tell Pillow to resize 
    resize_shape_wh = (model_input_shape[3], model_input_shape[2])
    test_image = Image.open(test_image_path)
    test_image = test_image.resize(resize_shape_wh, resample=Image.Resampling.BICUBIC)
    test_image = np.array(test_image, dtype=np.float32, order="C")

    # Reshape the np array that used to be a PIL image into "CHW" (PyTorch shape
    # convention).
    test_image = np.transpose(test_image, [2, 0, 1])
    # CHW to NCHW format. I.e., adding another dimension in case you had a batch
    # size greater than 1.
    test_image = np.expand_dims(test_image, axis=0)
    # Convert the image to row major order in memory.
    test_image = np.array(test_image, dtype=np.float32, order="C")

    
    # Set up runtime and deserialize the engine file (the TensorRT model
    f = open(trt_model_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate memory for input and output.
    inputs, outputs, bindings, stream = allocate_buffers(engine=engine)
    # Inputs is like a pointer to a collection of GPU memory locations. Likewise
    # for outputs. To run inference on our model and get the results, we
    # essentially have to load our image data from system memory into one of
    # those GPU memory input locations, then get the results in one of the
    # output GPU memory locations.

    # Also, we create what's called a CUDA stream object. A CUDA stream is kind
    # of like an OS thread--it's an independent sequence of (CUDA) instructions.
    # For our case, we only need one stream, as we only want the capability to
    # process a single image at a time. BUT, you could create multiple, parallel
    # streams to process multiple images asynchronously if you needed to. To add
    # instructions to this stream, we can specify this stream as an argument to
    # each of the CUDA operations that we specify below.

    # Pass image through model.
    print(f"Running inference on {test_image_path}")
    inputs[0].host = test_image

    # Transfer input image from the "host to the device" (htod), where the
    # device is the GPU.
    


if __name__ == "__main__":

    main()

    