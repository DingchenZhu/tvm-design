"""
Target hardware specification for VIS VPU.
"""

from typing import Dict, Any


class VISVPUTarget:
    """VIS VPU hardware target specification."""
    
    # Hardware capabilities
    MAX_LINE_BUFFERS = 2
    MAX_WEIGHT_BUFFERS = 3
    MAX_ACC_REGISTERS = 2
    
    # Memory hierarchy
    ON_CHIP_MEMORY_SIZE = 1024 * 1024  # 1MB
    LINE_BUFFER_SIZE = 256 * 1024      # 256KB per buffer
    
    # Compute capabilities
    SUPPORTED_DTYPES = ['float32', 'float16', 'int8', 'uint8']
    MAX_CONV_KERNEL_SIZE = 7
    MAX_CHANNELS_PER_TILE = 64
    
    # Instruction set
    SUPPORTED_OPS = {
        'nn.conv2d',
        'nn.conv2d_transpose',
        'nn.dense',
        'nn.relu',
        'nn.prelu',
        'nn.leaky_relu',
        'nn.avg_pool2d',
        'nn.max_pool2d',
        'add',
        'multiply',
        'sigmoid',
        'tanh',
    }
    
    @classmethod
    def get_target_dict(cls) -> Dict[str, Any]:
        """Get target as dictionary for TVM."""
        return {
            'kind': 'vis_vpu',
            'keys': ['vis_vpu'],
            'on_chip_memory': cls.ON_CHIP_MEMORY_SIZE,
        }
    
    @classmethod
    def is_supported_op(cls, op_name: str) -> bool:
        """Check if operator is supported."""
        return op_name in cls.SUPPORTED_OPS


# Create TVM target
def create_tvm_target():
    """Create TVM target object."""
    import tvm
    # For now, use LLVM as base target
    # In full implementation, would register custom VIS VPU target
    return tvm.target.Target("llvm")

