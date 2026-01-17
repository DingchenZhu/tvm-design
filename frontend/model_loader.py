"""
Unified model loader for multiple frameworks.

Provides a single interface to import models from PyTorch, ONNX, and other frameworks.
"""

import os
from typing import Union, Tuple, Dict, Optional, Any, List
from pathlib import Path
import logging
import tvm
from tvm import relay
import torch
import torch.nn as nn

from .onnx_importer import ONNXImporter
from .pytorch_importer import PyTorchImporter

logger = logging.getLogger(__name__)


class ModelLoader:
    """Unified interface for loading models from different frameworks."""
    
    SUPPORTED_FORMATS = ['pytorch', 'onnx', 'torchscript']
    
    def __init__(self, freeze_params: bool = True):
        """
        Initialize model loader.
        
        Args:
            freeze_params: Whether to freeze parameters in the graph
        """
        self.freeze_params = freeze_params
        self.onnx_importer = ONNXImporter(freeze_params=freeze_params)
        self.pytorch_importer = PyTorchImporter(freeze_params=freeze_params)
    
    def detect_format(self, model_or_path: Union[str, nn.Module]) -> str:
        """
        Automatically detect model format.
        
        Args:
            model_or_path: Model object or path to model file
            
        Returns:
            Model format string
        """
        if isinstance(model_or_path, nn.Module):
            return 'pytorch'
        elif isinstance(model_or_path, str):
            path = Path(model_or_path)
            ext = path.suffix.lower()
            
            if ext == '.onnx':
                return 'onnx'
            elif ext in ['.pt', '.pth']:
                return 'torchscript'
            else:
                raise ValueError(f"Unknown model format for file: {model_or_path}")
        else:
            raise ValueError(f"Unsupported model type: {type(model_or_path)}")
    
    def load(
        self,
        model_or_path: Union[str, nn.Module],
        model_format: Optional[str] = None,
        input_shapes: Optional[Union[Tuple, List[Tuple]]] = None,
        example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        input_names: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """
        Load model from any supported format.
        
        Args:
            model_or_path: PyTorch model object or path to model file
            model_format: Model format ('pytorch', 'onnx', 'torchscript')
                         Auto-detected if None
            input_shapes: Input shapes for the model (required for some formats)
            example_inputs: Example inputs for PyTorch models
            input_names: Optional input names
            **kwargs: Additional format-specific arguments
            
        Returns:
            Tuple of (Relay IR module, parameters dictionary)
        """
        # Detect format if not specified
        if model_format is None:
            model_format = self.detect_format(model_or_path)
        
        model_format = model_format.lower()
        if model_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported model format: {model_format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        logger.info(f"Loading model in '{model_format}' format")
        
        # Route to appropriate importer
        if model_format == 'onnx':
            return self._load_onnx(model_or_path, input_shapes, **kwargs)
        elif model_format == 'pytorch':
            return self._load_pytorch(model_or_path, example_inputs, input_names, **kwargs)
        elif model_format == 'torchscript':
            return self._load_torchscript(model_or_path, input_shapes, input_names, **kwargs)
        else:
            raise ValueError(f"Format '{model_format}' not yet implemented")
    
    def _load_onnx(
        self,
        model_path: str,
        input_shapes: Optional[Dict[str, Tuple]] = None,
        **kwargs
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """Load ONNX model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        shape_dict = input_shapes if isinstance(input_shapes, dict) else None
        dtype_dict = kwargs.get('dtype_dict', None)
        
        mod, params = self.onnx_importer.import_model(
            model_path,
            shape_dict=shape_dict,
            dtype_dict=dtype_dict
        )
        
        return mod, params
    
    def _load_pytorch(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        input_names: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """Load PyTorch model."""
        if example_inputs is None:
            raise ValueError("example_inputs is required for PyTorch models")
        
        strict_trace = kwargs.get('strict_trace', False)
        
        mod, params = self.pytorch_importer.import_model(
            model,
            example_inputs,
            input_names=input_names,
            strict_trace=strict_trace
        )
        
        return mod, params
    
    def _load_torchscript(
        self,
        model_path: str,
        input_shapes: Optional[List[Tuple]] = None,
        input_names: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """Load TorchScript model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")
        
        if input_shapes is None:
            raise ValueError("input_shapes is required for TorchScript models")
        
        # Load TorchScript
        scripted_model = self.pytorch_importer.load_torchscript(model_path)
        
        # Convert to Relay
        mod, params = self.pytorch_importer.convert_to_relay(
            scripted_model,
            input_shapes,
            input_names
        )
        
        return mod, params
    
    def visualize(
        self,
        mod: tvm.ir.IRModule,
        output_path: Optional[str] = None
    ) -> str:
        """
        Visualize Relay graph.
        
        Args:
            mod: Relay IR module
            output_path: Optional path to save visualization
            
        Returns:
            String representation of the graph
        """
        return self.onnx_importer.visualize_graph(mod, output_path)
    
    def print_stats(
        self,
        mod: tvm.ir.IRModule,
        params: Dict[str, tvm.nd.NDArray]
    ):
        """
        Print model statistics.
        
        Args:
            mod: Relay IR module
            params: Parameters dictionary
        """
        self.onnx_importer.print_graph_stats(mod, params)


def load_model(
    model_or_path: Union[str, nn.Module],
    model_format: Optional[str] = None,
    input_shapes: Optional[Union[Tuple, Dict[str, Tuple], List[Tuple]]] = None,
    example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    freeze_params: bool = True,
    **kwargs
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Convenience function to load a model.
    
    Args:
        model_or_path: PyTorch model object or path to model file
        model_format: Model format ('pytorch', 'onnx', 'torchscript')
        input_shapes: Input shapes for the model
        example_inputs: Example inputs for PyTorch models
        freeze_params: Whether to freeze parameters
        **kwargs: Additional format-specific arguments
        
    Returns:
        Tuple of (Relay IR module, parameters dictionary)
    """
    loader = ModelLoader(freeze_params=freeze_params)
    return loader.load(
        model_or_path,
        model_format=model_format,
        input_shapes=input_shapes,
        example_inputs=example_inputs,
        **kwargs
    )


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = ModelLoader()
    
    # Test ONNX loading
    print("\n" + "="*60)
    print("Testing ONNX Model Loading")
    print("="*60)
    try:
        mod, params = loader.load('../USR_Net.onnx', model_format='onnx')
        print("\n✓ Successfully loaded USR_Net.onnx")
        loader.print_stats(mod, params)
    except Exception as e:
        print(f"✗ Failed to load ONNX model: {e}")
    
    # Test PyTorch loading
    print("\n" + "="*60)
    print("Testing PyTorch Model Loading")
    print("="*60)
    try:
        from models_new_930 import FSRCNN
        
        model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
        example_input = torch.randn(1, 1, 270, 480)
        
        mod, params = loader.load(
            model,
            model_format='pytorch',
            example_inputs=example_input
        )
        print("\n✓ Successfully loaded FSRCNN")
        loader.print_stats(mod, params)
    except Exception as e:
        print(f"✗ Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()

