"""Contains utility functions for converting F1Tenth YOLO PyTorch model to
TensorRT.  ore specifically, provides functions for converting between
torch<-->ONNX, ONNX<-->TensorRT. Note that there are probably other command line
utilties out there to do this. torchTRT might do this, I bet roboflow's
supervision library has methods for doing this as well. TAO toolkit almost
certainly would have a utilty to do this already.

Was thinking we could solely use TorchTRT to do this, but that's a little
different. TorchTRT basically wraps using TensorRT in the Torch ecosystem. That
way, you still get better performance, but it's still not going to be as
performant as converting directly to a TensorRT engine and running that on its
own.
""" 

from pathlib import Path
from typing import Optional, Tuple

import torch.onnx
from torch import nn

from f1yolo import F110_YOLO

def f1tenth_torch_to_onnx(model_weights_path: Path,
                          input_size: Tuple,
                          batch_size: Optional[int] = 1,
                          output_directory: Optional[Path] = Path.cwd()) -> None:

    # Check to see if the model exists at the provided path.
    if not model_weights_path.exists():
        raise FileNotFoundError(f"PyTorch model {model_weights_path} could not be found!")
    if not output_directory.exists():
        raise Exception(f"Provided output directory {output_directory} could not be found!")

    # Create new instance of the model. Try to load weights and add them to the
    # model.
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    f1yolo_model = F110_YOLO()
    state_dict = torch.load(f=model_weights_path)
    f1yolo_model.load_state_dict(state_dict)

    # Convert to onnx. 
    # https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#export-from-pytorch
    # Create a dummy input with the expected input size. Will match the size of
    # the first layer.
    # TODO Need to add batch size as first dimension.
    # Note that input shape is (B, C, H, W)
    dummy_input = torch.zeros(size=(batch_size, *input_size))
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Also, create a new path within the provided output directory.
    onnx_model_path = output_directory/r"f1yolo.onnx"

    # https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
    torch.onnx.export(model=f1yolo_model,
                      args=dummy_input,
                      f=onnx_model_path,
                      verbose=True)
    
    return

# Note that this operation can very easily be carried out via the trtexec
# command line utility provided as a part of the TensorRT SDK. This function is
# an example of converting an ONNX model to a TensorRT engine using the TensorRT
# python api/library.
def onnx_to_tensorrt() -> None:

    pass

if __name__ == "__main__":
    
    # 1. Convert model to ONNX.
    model_weights_path = Path(r"./f1yolo.pt")
    f1tenth_torch_to_onnx(model_weights_path=model_weights_path,
                          input_size=(3, 180, 320))