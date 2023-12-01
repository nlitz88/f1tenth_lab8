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
import torch

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

# Adapted from
# https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
def allocate_buffers(engine):
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            input_dtype = trt.nptype(engine.get_tensor_dtype(binding))
            host_input = cuda.pagelocked_empty(input_size, input_dtype)
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    return host_input, device_input, host_output, device_output

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

    f1yolo_engine = runtime.deserialize_cuda_engine(f.read())
    context = f1yolo_engine.create_execution_context()

    # Allocate memory for input and output.
    host_input, device_input, host_output, device_output = allocate_buffers(engine=f1yolo_engine)
    # host_input is a region of page_locked memory on the host, while
    # device_input is a memory location allocated on the GPU. Likewise,
    # host_output is a region of page_locked memory on the host, and
    # device_output is the memory region on the device that is allocated to
    # store the output of the engine.

    # Also, we create what's called a CUDA stream object. A CUDA stream is kind
    # of like an OS thread--it's an independent sequence of (CUDA) instructions.
    # For our case, we only need one stream, as we only want the capability to
    # process a single image at a time. BUT, you could create multiple, parallel
    # streams to process multiple images asynchronously if you needed to. To add
    # instructions to this stream, we can specify this stream as an argument to
    # each of the CUDA operations that we specify below.
    inference_stream = cuda.Stream()

    # Pass image through model.
    print(f"Running inference on {test_image_path}")
    # Copy the input image into the host_input page_locked memory.
    host_input = test_image
    # Copy the image data in the hosts memory to the device's memory.
    cuda.memcpy_htod_async(device_input, host_input, inference_stream)
    # Run inference. This essentially tells teh CPU to take the input and pass
    # it through the engine, place the result in the output location.
    context.execute_async(bindings=[int(device_input), int(device_output)],
                          stream_handle=inference_stream.handle)
    # Copy the results from GPU memory back to the host.
    cuda.memcpy_dtoh_async(host_output, device_output, inference_stream)
    inference_stream.synchronize()

    # Examine dimensions of output data, then we can figure out how to reshape
    # it before post-processing.
    result_tensor = torch.Tensor(host_output)
    print(f"Result vector has shape: {result_tensor.shape}")

if __name__ == "__main__":

    main()

    