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
import subprocess

import torch.onnx
from torch import nn

from f1yolo import F110_YOLO

def f1tenth_torch_to_onnx(model_weights_path: Path,
                          input_size: Tuple,
                          batch_size: Optional[int] = 1,
                          output_directory: Optional[Path] = Path.cwd()) -> None:
    """Converts the weights of the provided F1Tenth YOLO model to ONNX format.

    Args:
        model_weights_path (Path): Path to the model weights in torch format
        (I.e., a ".pt" file).
        input_size (Tuple): The dimensions of a single input to the network in
        order (C,H,W) (Torch order).
        batch_size (Optional[int], optional): The number of images that should
        be passed forward through the network at a time. Defaults to 1.
        output_directory (Optional[Path], optional): Directory the onnx model
        should be stored in. Defaults to Path.cwd() (directory you run script
        from).

    Raises:
        FileNotFoundError: Thrown if the provided weights file doesn't exist.
        Exception: Thrown in the provided output directory doesn't exist.
    
    Returns:
        str: The path of the resulting onnx file if successfully created.
    """

    # Check to see if the model exists at the provided path.
    if not model_weights_path.exists():
        raise FileNotFoundError(f"PyTorch model {model_weights_path} could not be found!")
    if not output_directory.exists():
        raise Exception(f"Provided output directory {output_directory} could not be found!")

    # Create new instance of the model. Try to load weights and add them to the
    # model.
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    f1yolo_model = F110_YOLO()
    print(f"Loading model weights.")
    state_dict = torch.load(f=model_weights_path)
    f1yolo_model.load_state_dict(state_dict)
    print(f"Loaded model weights.")

    # Convert to onnx. 
    # https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#export-from-pytorch
    # Create a dummy input with the expected input size. Will match the size of
    # the first layer.
    # TODO Need to add batch size as first dimension.
    # Note that input shape is (B, C, H, W)
    dummy_input = torch.zeros(size=(batch_size, *input_size))

    # Also, create a new path within the provided output directory.
    onnx_model_path = output_directory/f"{Path(model_weights_path.parts[-1]).stem}.onnx"
    print(f"Onnx model path: {onnx_model_path}")

    # https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
    print(f"Beginning onnx conversion.")
    torch.onnx.export(model=f1yolo_model,
                      args=dummy_input,
                      f=onnx_model_path,
                      verbose=False)
    print(f"Finished onnx conversion.")
    
    return onnx_model_path

# Note that this operation can very easily be carried out via the trtexec
# command line utility provided as a part of the TensorRT SDK. For the sake of
# simplicity, I'm going to use the trtexec command line utility. Was originally
# going to use the Python API to do the conversion, but there doesn't appear to
# be a ton of documentation for it.

# Following this guide:
# https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#convert-onnx-engine
def onnx_to_tensorrt(onnx_model_path: Path,
                     quantization: str) -> None:
    """Converts the provided onnx model to a TensorRT engine.

    Args:
        onnx_model_path (Path): Filepath of the onnx model to convert
        quantization (str): Precision that the resulting engine/model should
        have. Choose from ["fp32", "fp16", "fp8", "int8"].

    Raises:
        Exception: Throws exception if an invalid precision/quantization
        provided.
    """
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags

    if quantization == "fp32":
        subprocess.run(args=["trtexec", f"--onnx={onnx_model_path}", f"--saveEngine={Path(onnx_model_path.parts[-1]).stem}_{quantization}.trt"])
    elif quantization == "fp16" or quantization == "int8" or quantization == "fp8":
        subprocess.run(args=["trtexec", f"--onnx={onnx_model_path}", f"--{quantization}", f"--saveEngine={Path(onnx_model_path.parts[-1]).stem}_{quantization}.trt"])
    else:
        raise Exception(f"Invalid/unsupported quantization/precision level {quantization} provided. Please choose from fp32, fp16, fp8, int8")
    return

if __name__ == "__main__":
    
    # 1. Convert model to onnx format.
    model_weights_path = Path(r"./f1yolo.pt")
    onnx_model_path = f1tenth_torch_to_onnx(model_weights_path=model_weights_path,
                                            input_size=(3, 180, 320))
    
    # 2. Convert the onnx model to a TensorRT Engine.
    onnx_to_tensorrt(onnx_model_path=onnx_model_path,
                     quantization="int8")