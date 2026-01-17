"""
ONNX model importer for TVM Relay.

Converts ONNX models to Relay IR with proper shape inference and type checking.
"""

import onnx
from onnx import numpy_helper
import numpy as np
from typing import Dict, Tuple, Optional, Any
import tvm
from tvm import relay
from tvm.relay import transform
import logging

logger = logging.getLogger(__name__)


class ONNXImporter:
    """Import ONNX models to TVM Relay IR."""
    
    def __init__(self, freeze_params: bool = True, opset: Optional[int] = None):
        """
        Initialize ONNX importer.
        
        Args:
            freeze_params: Whether to freeze parameters in the graph
            opset: ONNX opset version (auto-detected if None)
        """
        self.freeze_params = freeze_params
        self.opset = opset
        
    def load_model(self, model_path: str) -> onnx.ModelProto:
        """
        Load ONNX model from file.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            ONNX model proto
        """
        logger.info(f"Loading ONNX model from {model_path}")
        model = onnx.load(model_path)
        
        # Check and simplify model
        try:
            onnx.checker.check_model(model)
            logger.info("ONNX model is valid")
        except Exception as e:
            logger.warning(f"ONNX model validation warning: {e}")
        
        return model
    
    def get_input_info(self, model: onnx.ModelProto) -> Dict[str, Tuple]:
        """
        Extract input information from ONNX model.
        
        Args:
            model: ONNX model proto
            
        Returns:
            Dictionary mapping input names to (shape, dtype) tuples
        """
        input_info = {}
        
        for input_tensor in model.graph.input:
            # Skip if it's an initializer (i.e., a parameter)
            if input_tensor.name in [init.name for init in model.graph.initializer]:
                continue
                
            name = input_tensor.name
            shape = []
            
            # Extract shape
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(int(dim.dim_value))
                elif dim.HasField('dim_param'):
                    # Dynamic dimension, use -1
                    shape.append(-1)
                else:
                    shape.append(-1)
            
            # Extract dtype
            dtype_map = {
                1: 'float32',   # FLOAT
                2: 'uint8',     # UINT8
                3: 'int8',      # INT8
                6: 'int32',     # INT32
                7: 'int64',     # INT64
                10: 'float16',  # FLOAT16
                11: 'float64',  # DOUBLE
            }
            
            dtype_id = input_tensor.type.tensor_type.elem_type
            dtype = dtype_map.get(dtype_id, 'float32')
            
            input_info[name] = (tuple(shape), dtype)
            logger.info(f"Input '{name}': shape={shape}, dtype={dtype}")
        
        return input_info
    
    def get_output_info(self, model: onnx.ModelProto) -> Dict[str, Tuple]:
        """
        Extract output information from ONNX model.
        
        Args:
            model: ONNX model proto
            
        Returns:
            Dictionary mapping output names to (shape, dtype) tuples
        """
        output_info = {}
        
        for output_tensor in model.graph.output:
            name = output_tensor.name
            shape = []
            
            # Extract shape
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(int(dim.dim_value))
                elif dim.HasField('dim_param'):
                    shape.append(-1)
                else:
                    shape.append(-1)
            
            # Extract dtype
            dtype_map = {
                1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32',
                7: 'int64', 10: 'float16', 11: 'float64',
            }
            
            dtype_id = output_tensor.type.tensor_type.elem_type
            dtype = dtype_map.get(dtype_id, 'float32')
            
            output_info[name] = (tuple(shape), dtype)
            logger.info(f"Output '{name}': shape={shape}, dtype={dtype}")
        
        return output_info
    
    def convert_to_relay(
        self, 
        model: onnx.ModelProto,
        shape_dict: Optional[Dict[str, Tuple]] = None,
        dtype_dict: Optional[Dict[str, str]] = None
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """
        Convert ONNX model to Relay IR.
        
        Args:
            model: ONNX model proto
            shape_dict: Optional dictionary of input shapes (overrides model)
            dtype_dict: Optional dictionary of input dtypes (overrides model)
            
        Returns:
            Tuple of (Relay IR module, parameters dictionary)
        """
        logger.info("Converting ONNX model to Relay IR")
        
        # Get input information
        input_info = self.get_input_info(model)
        
        # Build shape dictionary
        if shape_dict is None:
            shape_dict = {}
            for name, (shape, dtype) in input_info.items():
                shape_dict[name] = shape
        
        # Build dtype dictionary
        if dtype_dict is None:
            dtype_dict = {}
            for name, (shape, dtype) in input_info.items():
                dtype_dict[name] = dtype
        
        # Convert to Relay
        try:
            mod, params = relay.frontend.from_onnx(
                model,
                shape=shape_dict,
                dtype=dtype_dict,
                opset=self.opset
            )
            logger.info("Successfully converted ONNX to Relay IR")
            logger.info(f"Number of parameters: {len(params)}")
            
            return mod, params
            
        except Exception as e:
            logger.error(f"Failed to convert ONNX to Relay: {e}")
            raise
    
    def import_model(
        self,
        model_path: str,
        shape_dict: Optional[Dict[str, Tuple]] = None,
        dtype_dict: Optional[Dict[str, str]] = None
    ) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
        """
        Complete import pipeline: load ONNX model and convert to Relay.
        
        Args:
            model_path: Path to ONNX model file
            shape_dict: Optional dictionary of input shapes
            dtype_dict: Optional dictionary of input dtypes
            
        Returns:
            Tuple of (Relay IR module, parameters dictionary)
        """
        # Load model
        model = self.load_model(model_path)
        
        # Convert to Relay
        mod, params = self.convert_to_relay(model, shape_dict, dtype_dict)
        
        # Apply basic cleanup passes
        if self.freeze_params and params:
            logger.info("Freezing parameters in the graph")
            with tvm.transform.PassContext(opt_level=0):
                mod = relay.transform.InferType()(mod)
        
        return mod, params
    
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
        # Count operators
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


def import_onnx_model(
    model_path: str,
    shape_dict: Optional[Dict[str, Tuple]] = None,
    dtype_dict: Optional[Dict[str, str]] = None,
    freeze_params: bool = True
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Convenience function to import ONNX model.
    
    Args:
        model_path: Path to ONNX model file
        shape_dict: Optional dictionary of input shapes
        dtype_dict: Optional dictionary of input dtypes
        freeze_params: Whether to freeze parameters
        
    Returns:
        Tuple of (Relay IR module, parameters dictionary)
    """
    importer = ONNXImporter(freeze_params=freeze_params)
    return importer.import_model(model_path, shape_dict, dtype_dict)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Import USR_Net
    model_path = '../USR_Net.onnx'
    
    try:
        importer = ONNXImporter()
        mod, params = importer.import_model(model_path)
        
        # Print statistics
        importer.print_graph_stats(mod, params)
        
        # Save graph visualization
        importer.visualize_graph(mod, 'usr_net_relay_graph.txt')
        
        print("\nSuccessfully imported USR_Net!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

