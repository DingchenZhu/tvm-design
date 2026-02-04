"""
Bridge between TVM Relay IR and VIS VPU instruction generator.

This module traverses the optimized Relay IR and calls the appropriate
instruction generation functions to produce hardware instructions.
"""

import sys
sys.path.append('..')

import tvm
from tvm import relay
from typing import Dict, List, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RelayToInstructionBridge:
    """Bridge from Relay IR to VIS VPU instructions."""
    
    def __init__(self):
        """Initialize the bridge."""
        self.instructions = []
        self.layer_info = []
        self.current_layer_idx = 0
        
    def traverse_and_generate(self, mod: tvm.IRModule, params: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Traverse Relay IR and generate instructions.
        
        Args:
            mod: TVM IRModule containing Relay IR
            params: Model parameters
            
        Returns:
            List of instruction dictionaries
        """
        logger.info("Starting IR traversal for instruction generation...")
        
        # Get the main function
        func = mod["main"]
        
        # Extract operators from the IR
        self._extract_layers(func, params)
        
        # Generate instructions for each layer
        self._generate_instructions_from_layers()
        
        logger.info(f"Generated instructions for {len(self.layer_info)} layers")
        
        return self.instructions
    
    def _extract_layers(self, func: relay.Function, params: Dict):
        """Extract layer information from Relay function."""
        logger.info("Extracting layers from Relay IR...")
        
        class LayerExtractor(relay.ExprVisitor):
            """Visitor to extract layer information."""
            
            def __init__(self, params):
                super().__init__()
                self.layers = []
                self.params = params
                self.layer_idx = 0
                
            def visit_call(self, call):
                """Visit call nodes to extract operators."""
                if isinstance(call.op, tvm.ir.Op):
                    op_name = call.op.name
                    
                    # Extract operator information
                    layer_info = {
                        'idx': self.layer_idx,
                        'op': op_name,
                        'attrs': {},
                        'input_shapes': [],
                        'output_shape': None
                    }
                    
                    # Get attributes
                    if hasattr(call, 'attrs') and call.attrs:
                        for key in call.attrs.keys():
                            layer_info['attrs'][key] = getattr(call.attrs, key)
                    
                    # Get input shapes
                    for arg in call.args:
                        if hasattr(arg, 'checked_type'):
                            shape = [int(d) for d in arg.checked_type.shape]
                            layer_info['input_shapes'].append(shape)
                    
                    # Get output shape
                    if hasattr(call, 'checked_type'):
                        layer_info['output_shape'] = [int(d) for d in call.checked_type.shape]
                    
                    self.layers.append(layer_info)
                    self.layer_idx += 1
                    
                    logger.debug(f"Layer {self.layer_idx}: {op_name}")
                
                # Continue traversal
                super().visit_call(call)
        
        extractor = LayerExtractor(params)
        extractor.visit(func)
        self.layer_info = extractor.layers
        
        logger.info(f"Extracted {len(self.layer_info)} layers")
        
    def _generate_instructions_from_layers(self):
        """Generate instructions from extracted layer information."""
        logger.info("Generating instructions from layers...")
        
        # Import instruction API
        from instruction import (
            DataLoader, WeightLoader, DataStorer, 
            QuantLoader, OffchipDataLoader
        )
        
        # For now, this is a simplified version
        # You would call your sd_sr_codegen functions here
        # based on the layer types and configurations
        
        for layer in self.layer_info:
            op_name = layer['op']
            
            if 'conv2d' in op_name:
                self._generate_conv2d_instructions(layer)
            elif 'dense' in op_name or 'matmul' in op_name:
                self._generate_dense_instructions(layer)
            elif 'pool' in op_name:
                self._generate_pool_instructions(layer)
            else:
                logger.warning(f"No instruction generator for {op_name}")
    
    def _generate_conv2d_instructions(self, layer: Dict):
        """Generate instructions for Conv2D layer."""
        from instruction import DataLoader, WeightLoader, DataStorer, QuantLoader
        
        logger.info(f"Generating Conv2D instructions for layer {layer['idx']}")
        
        # Extract layer parameters
        input_shape = layer['input_shapes'][0] if layer['input_shapes'] else [1, 3, 224, 224]
        output_shape = layer['output_shape'] if layer['output_shape'] else [1, 64, 112, 112]
        
        attrs = layer['attrs']
        
        # This is a simplified example - you would customize based on
        # your hardware constraints and the specific layer configuration
        
        # 1. Load quantization parameters
        QuantLoader.dispatch(
            quant_reg_load_idx=0,
            quant_mode=0,
            layer_idx=layer['idx'],
            transnum=4,
            bas_addr=0
        )
        
        # 2. Load data
        DataLoader.dispatch(
            layer_idx=layer['idx'],
            line_buffer_reshape=0,
            is_padding_row=0,
            read_mode=0,
            transnum=input_shape[-1] if len(input_shape) > 0 else 224,
            line_buffer_idx=0,
            src_buffer_idx='a',
            bas_addr=0
        )
        
        # 3. Load weights
        WeightLoader.dispatch(
            acc_reg_comp_idx=0,
            kernal_size=0,
            line_buffer_row_shift=1,
            line_buffer_idx=0,
            is_padding_col=1,
            weight_parall_mode=0,
            is_new=0,
            transnum=9,
            bas_addr=0,
            is_bilinear_bicubic=0,
            offset_reg_idx=0
        )
        
        # 4. Store results
        DataStorer.dispatch(
            quant_config_idx=0,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=0,
            acc_mode=0,
            transfer_num=1,
            store_mode=0,
            stride=224,
            base_addr_pooling=0,
            base_addrs_res=0,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx='b'
        )
    
    def _generate_dense_instructions(self, layer: Dict):
        """Generate instructions for Dense layer."""
        logger.info(f"Generating Dense instructions for layer {layer['idx']}")
        # Similar to Conv2D
        
    def _generate_pool_instructions(self, layer: Dict):
        """Generate instructions for Pooling layer."""
        logger.info(f"Generating Pooling instructions for layer {layer['idx']}")
        # Pooling is usually fused with conv in DataStorer


def generate_instructions_from_relay(mod: tvm.IRModule, params: Dict[str, np.ndarray]) -> List[Dict]:
    """
    Main entry point to generate instructions from Relay IR.
    
    Args:
        mod: TVM IRModule
        params: Model parameters
        
    Returns:
        List of instruction dictionaries
    """
    bridge = RelayToInstructionBridge()
    return bridge.traverse_and_generate(mod, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("IR to Instruction Bridge")
    print("This module connects TVM Relay IR to VIS VPU instructions")

