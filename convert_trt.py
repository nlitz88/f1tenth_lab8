"""Contains utility functions for converting PyTorch models to TensorRT
models.
""" 

from pathlib import Path

import torch.onnx

def torch_to_onnx(torch_model_path: Path,
                  output_directory: Path) -> None:
    
    # Attempt to load model.
    pass