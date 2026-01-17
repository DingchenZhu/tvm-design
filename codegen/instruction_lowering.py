"""
Lower Relay operators to VIS VPU hardware instructions.

Maps high-level operators to instruction sequences using the
instruction API from instruction.py.
"""

import sys
sys.path.append('..')

from instruction import *
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class InstructionLowering:
    """Lower operators to VIS VPU instructions."""
    
    def __init__(self):
        """Initialize instruction lowering."""
        self.instructions = []
        self.layer_idx = 0
        self.buffer_state = {
            'data_buffer': 'a',  # Current data buffer (a or b)
            'line_buffer': 0,     # Current line buffer (0 or 1)
            'acc_reg': 0,         # Current accumulator register
        }
    
    def lower_conv2d(self, attrs: Dict[str, Any], input_shape: Tuple, weight_shape: Tuple) -> List[Dict]:
        """
        Lower Conv2D to instruction sequence.
        
        Args:
            attrs: Conv2D attributes (stride, padding, etc.)
            input_shape: Input tensor shape (N, C, H, W)
            weight_shape: Weight tensor shape (OC, IC, KH, KW)
            
        Returns:
            List of instruction dictionaries
        """
        N, IC, IH, IW = input_shape
        OC, _, KH, KW = weight_shape
        
        stride = attrs.get('strides', (1, 1))[0]
        padding = attrs.get('padding', (1, 1))[0]
        
        OH = (IH + 2 * padding - KH) // stride + 1
        OW = (IW + 2 * padding - KW) // stride + 1
        
        logger.info(f"Lowering Conv2D: {input_shape} -> {(N, OC, OH, OW)}")
        
        # Generate instruction sequence
        inst_seq = []
        
        # Data loader instructions
        for h in range(OH):
            DataLoader.dispatch(
                layer_idx=self.layer_idx,
                line_buffer_reshape=0,
                is_padding_row=0,
                read_mode=0,
                transnum=IW,
                line_buffer_idx=self.buffer_state['line_buffer'],
                src_buffer_idx=self.buffer_state['data_buffer'],
                bas_addr=h * IW * IC
            )
            
            # Weight loader
            WeightLoader.dispatch(
                acc_reg_comp_idx=self.buffer_state['acc_reg'],
                kernal_size=KH,
                line_buffer_row_shift=0,
                line_buffer_idx=self.buffer_state['line_buffer'],
                is_padding_col=0,
                weight_parall_mode=0,
                is_new=1 if h == 0 else 0,
                transnum=OC,
                bas_addr=0,
                is_bilinear_bicubic=0,
                offset_reg_idx=0
            )
            
            # Data storer
            DataStorer.dispatch(
                quant_config_idx=0,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=False,
                pooling_out_mode=0,
                pooling_out_new=False,
                is_pooling=False,
                reg_out_idx=self.buffer_state['acc_reg'],
                acc_mode=0,
                transfer_num=OW,
                store_mode=0,
                stride=stride,
                base_addr_pooling=0,
                base_addrs_res=h * OW * OC,
                is_bicubic_add=False,
                is_first_or_last_row=0,
                is_mask=False,
                is_new=True,
                dest_buffer_idx='b' if self.buffer_state['data_buffer'] == 'a' else 'a'
            )
        
        # Update state
        self._switch_buffers()
        self.layer_idx += 1
        
        return Inst.code_list[-len(inst_seq):]
    
    def lower_relu(self, input_shape: Tuple) -> List[Dict]:
        """Lower ReLU activation."""
        logger.info(f"Lowering ReLU: {input_shape}")
        # ReLU is typically fused into previous conv/dense
        # If standalone, implement as clip(x, 0, inf)
        return []
    
    def lower_pool2d(self, attrs: Dict[str, Any], input_shape: Tuple, pool_type: str = 'avg') -> List[Dict]:
        """Lower pooling operation."""
        logger.info(f"Lowering {pool_type}_pool2d: {input_shape}")
        # Pooling is handled by DataStorer with pooling mode
        return []
    
    def lower_dense(self, attrs: Dict[str, Any], input_shape: Tuple, weight_shape: Tuple) -> List[Dict]:
        """Lower Dense (fully connected) operation."""
        logger.info(f"Lowering Dense: {input_shape} x {weight_shape}")
        # Similar to Conv2D but with 1x1 spatial dimensions
        return self.lower_conv2d(attrs, input_shape, weight_shape)
    
    def _switch_buffers(self):
        """Switch ping-pong buffers."""
        self.buffer_state['data_buffer'] = 'b' if self.buffer_state['data_buffer'] == 'a' else 'a'
        self.buffer_state['line_buffer'] = 1 - self.buffer_state['line_buffer']
        self.buffer_state['acc_reg'] = 1 - self.buffer_state['acc_reg']
    
    def get_instructions(self) -> List[Dict]:
        """Get all generated instructions."""
        return Inst.code_list
    
    def clear(self):
        """Clear instruction list."""
        Inst.code_list = []
        Inst.current_code_num = 0
        self.layer_idx = 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing instruction lowering...")
    
    lowering = InstructionLowering()
    
    # Test Conv2D lowering
    lowering.lower_conv2d(
        attrs={'strides': (1, 1), 'padding': (1, 1)},
        input_shape=(1, 3, 224, 224),
        weight_shape=(64, 3, 3, 3)
    )
    
    instructions = lowering.get_instructions()
    print(f"\nGenerated {len(instructions)} instructions")
    print(f"First instruction: {instructions[0] if instructions else 'None'}")
    
    print("\n✓ Instruction lowering test complete!")

