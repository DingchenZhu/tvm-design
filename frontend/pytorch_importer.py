"""
PyTorch model importer for TVM Relay.

Converts PyTorch models to Relay IR via TorchScript, with support for custom operators.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
import logging

logger = logging.getLogger(__name__)


class PyTorchImporter:
    """Import PyTorch models to TVM Relay IR."""
    
    def __init__(self, freeze_params: bool = True, use_script: bool = False):
        """
        Initialize PyTorch importer.
        
        Args:
            freeze_params: Whether to freeze parameters in the graph
            use_script: Use torch.jit.script instead of torch.jit.trace
        """
        self.freeze_params = freeze_params
        self.use_script = use_script
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare PyTorch model for export.
        
        Args:
            model: PyTorch model
            
        Returns:
            Prepared model in eval mode
        """
        logger.info("Preparing PyTorch model for export")
        
        # Set to eval mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def trace_model(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        strict: bool = False
    ) -> torch.jit.ScriptModule:
        """
        Trace PyTorch model to TorchScript.
        
        Args:
            model: PyTorch model
            example_inputs: Example input tensors for tracing
            strict: Whether to enforce strict tracing
            
        Returns:
            TorchScript module
        """
        logger.info("Tracing PyTorch model to TorchScript")
        
        # Prepare model
        model = self.prepare_model(model)
        
        # Trace or script
        if self.use_script:
            logger.info("Using torch.jit.script")
            try:
                scripted_model = torch.jit.script(model)
            except Exception as e:
                logger.warning(f"torch.jit.script failed: {e}")
                logger.info("Falling back to torch.jit.trace")
                scripted_model = torch.jit.trace(model, example_inputs, strict=strict)
        else:
            logger.info("Using torch.jit.trace")
            scripted_model = torch.jit.trace(model, example_inputs, strict=strict)
        
        # Optimize for inference
        scripted_model = torch.jit.freeze(scripted_model)
        
        return scripted_model
    
    def get_input_shapes(
        self,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> List[Tuple]:
        """
        Extract input shapes from example inputs.
        
        Args:
            example_inputs: Example input tensors
            
        Returns:
            List of input shapes
        """
        if isinstance(example_inputs, torch.Tensor):
            example_inputs = (example_inputs,)
        
        shapes = []
        for i, inp in enumerate(example_inputs):
            shape = tuple(inp.shape)
            logger.info(f"Input {i}: shape={shape}, dtype={inp.dtype}")
            shapes.append(shape)
        
        return shapes
    
    def convert_to_relay(
        self,
        scripted_model: torch.jit.ScriptModule,
        input_shapes: List[Tuple],
        input_names: Optional[List[str]] = None
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """
        Convert TorchScript model to Relay IR.
        
        Args:
            scripted_model: TorchScript module
            input_shapes: List of input shapes
            input_names: Optional list of input names
            
        Returns:
            Tuple of (Relay IR module, parameters dictionary)
        """
        logger.info("Converting TorchScript to Relay IR")
        
        # Build input info list
        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(input_shapes))]
        
        input_infos = []
        for name, shape in zip(input_names, input_shapes):
            input_infos.append((name, shape))
        
        logger.info(f"Input infos: {input_infos}")
        
        # Convert to Relay
        try:
            mod, params = relay.frontend.from_pytorch(
                scripted_model,
                input_infos
            )
            logger.info("Successfully converted PyTorch to Relay IR")
            logger.info(f"Number of parameters: {len(params)}")
            
            return mod, params
            
        except Exception as e:
            logger.error(f"Failed to convert PyTorch to Relay: {e}")
            raise
    
    def import_model(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        input_names: Optional[List[str]] = None,
        strict_trace: bool = False
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """
        Complete import pipeline: trace PyTorch model and convert to Relay.
        
        Args:
            model: PyTorch model
            example_inputs: Example input tensors for tracing
            input_names: Optional list of input names
            strict_trace: Whether to enforce strict tracing
            
        Returns:
            Tuple of (Relay IR module, parameters dictionary)
        """
        # Get input shapes
        input_shapes = self.get_input_shapes(example_inputs)
        
        # Trace model
        scripted_model = self.trace_model(model, example_inputs, strict=strict_trace)
        
        # Convert to Relay
        mod, params = self.convert_to_relay(scripted_model, input_shapes, input_names)
        
        # Apply basic cleanup passes
        if self.freeze_params and params:
            logger.info("Freezing parameters in the graph")
            with tvm.transform.PassContext(opt_level=0):
                mod = relay.transform.InferType()(mod)
        
        return mod, params
    
    def save_torchscript(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str
    ):
        """
        Save TorchScript model to file.
        
        Args:
            model: PyTorch model
            example_inputs: Example input tensors
            output_path: Path to save TorchScript model
        """
        scripted_model = self.trace_model(model, example_inputs)
        torch.jit.save(scripted_model, output_path)
        logger.info(f"Saved TorchScript model to {output_path}")
    
    def load_torchscript(self, model_path: str) -> torch.jit.ScriptModule:
        """
        Load TorchScript model from file.
        
        Args:
            model_path: Path to TorchScript model
            
        Returns:
            TorchScript module
        """
        logger.info(f"Loading TorchScript model from {model_path}")
        scripted_model = torch.jit.load(model_path)
        scripted_model.eval()
        return scripted_model
    
    def visualize_graph(self, mod: tvm.ir.IRModule, output_path: Optional[str] = None) -> str:
        """
        Visualize the Relay graph.
        
        Args:
            mod: Relay IR module
            output_path: Optional path to save visualization
            
        Returns:
            String representation of the graph
        """
        graph_str = mod.astext(show_meta_data=False)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(graph_str)
            logger.info(f"Saved graph visualization to {output_path}")
        
        return graph_str
    
    def print_graph_stats(self, mod: tvm.ir.IRModule, params: Dict[str, tvm.nd.NDArray]):
        """
        Print statistics about the imported graph.
        
        Args:
            mod: Relay IR module
            params: Parameters dictionary
        """
        from collections import Counter
        
        class OpCounter(relay.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.ops = []
            
            def visit_call(self, call):
                if isinstance(call.op, tvm.ir.Op):
                    self.ops.append(call.op.name)
                super().visit_call(call)
        
        counter = OpCounter()
        counter.visit(mod['main'])
        op_counts = Counter(counter.ops)
        
        logger.info("=== Graph Statistics ===")
        logger.info(f"Total operators: {len(counter.ops)}")
        logger.info(f"Unique operators: {len(op_counts)}")
        logger.info("\nOperator counts:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {op}: {count}")
        
        # Parameter stats
        total_params = 0
        total_size = 0
        for name, param in params.items():
            param_count = np.prod(param.shape)
            param_size = param_count * np.dtype(param.dtype).itemsize
            total_params += param_count
            total_size += param_size
        
        logger.info(f"\n=== Parameter Statistics ===")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Total size: {total_size / 1024 / 1024:.2f} MB")


def import_pytorch_model(
    model: nn.Module,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    input_names: Optional[List[str]] = None,
    freeze_params: bool = True
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Convenience function to import PyTorch model.
    
    Args:
        model: PyTorch model
        example_inputs: Example input tensors
        input_names: Optional list of input names
        freeze_params: Whether to freeze parameters
        
    Returns:
        Tuple of (Relay IR module, parameters dictionary)
    """
    importer = PyTorchImporter(freeze_params=freeze_params)
    return importer.import_model(model, example_inputs, input_names)


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('..')
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        from models_new_930 import FSRCNN
        
        # Create model
        model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 1, 270, 480)
        
        # Import model
        importer = PyTorchImporter()
        mod, params = importer.import_model(model, example_input)
        
        # Print statistics
        importer.print_graph_stats(mod, params)
        
        # Save graph visualization
        importer.visualize_graph(mod, 'fsrcnn_relay_graph.txt')
        
        print("\nSuccessfully imported FSRCNN!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

