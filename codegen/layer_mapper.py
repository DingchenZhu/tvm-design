"""
Map Relay IR operators to instruction sequences.
"""

import tvm
from tvm import relay
from typing import Dict, List, Any
import logging

from .instruction_lowering import InstructionLowering

logger = logging.getLogger(__name__)


class LayerMapper:
    """Map Relay operators to VIS VPU instructions."""
    
    def __init__(self):
        self.lowering = InstructionLowering()
        self.op_handlers = {
            'nn.conv2d': self._handle_conv2d,
            'nn.dense': self._handle_dense,
            'nn.relu': self._handle_relu,
            'nn.avg_pool2d': self._handle_avg_pool,
            'nn.max_pool2d': self._handle_max_pool,
        }
    
    def map_operator(self, op_name: str, attrs: Any, input_shapes: List) -> List[Dict]:
        """Map single operator to instructions."""
        handler = self.op_handlers.get(op_name)
        if handler:
            return handler(attrs, input_shapes)
        else:
            logger.warning(f"No handler for operator: {op_name}")
            return []
    
    def _handle_conv2d(self, attrs, input_shapes):
        return self.lowering.lower_conv2d(attrs, input_shapes[0], input_shapes[1])
    
    def _handle_dense(self, attrs, input_shapes):
        return self.lowering.lower_dense(attrs, input_shapes[0], input_shapes[1])
    
    def _handle_relu(self, attrs, input_shapes):
        return self.lowering.lower_relu(input_shapes[0])
    
    def _handle_avg_pool(self, attrs, input_shapes):
        return self.lowering.lower_pool2d(attrs, input_shapes[0], 'avg')
    
    def _handle_max_pool(self, attrs, input_shapes):
        return self.lowering.lower_pool2d(attrs, input_shapes[0], 'max')
    
    def get_all_instructions(self) -> List[Dict]:
        """Get all generated instructions."""
        return self.lowering.get_instructions()

