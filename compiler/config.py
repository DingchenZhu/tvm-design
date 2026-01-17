"""
Compiler configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import sys
sys.path.append('..')

from passes.pass_config import PassConfig, VIS_VPU_CONFIG


@dataclass
class CompilerConfig:
    """
    Configuration for the TVM VIS compiler.
    """
    
    # Target hardware
    target: str = 'vis_vpu'
    
    # Optimization configuration
    pass_config: PassConfig = field(default_factory=lambda: VIS_VPU_CONFIG)
    
    # Output configuration
    output_dir: str = './output'
    output_name: str = 'model'
    
    # Code generation options
    enable_scheduling: bool = True
    enable_tiling: bool = True
    
    # Memory configuration
    on_chip_memory_size: int = 1024 * 1024  # 1MB
    
    # Debug options
    dump_ir: bool = False
    dump_instructions: bool = True
    visualize_passes: bool = False
    
    # Model-specific options
    input_shapes: Optional[Dict[str, tuple]] = None
    input_dtypes: Optional[Dict[str, str]] = None


# Preset configurations
DEFAULT_CONFIG = CompilerConfig()

DEBUG_CONFIG = CompilerConfig(
    dump_ir=True,
    dump_instructions=True,
    visualize_passes=True,
)

FAST_COMPILE_CONFIG = CompilerConfig(
    pass_config=PassConfig(opt_level=1),
    enable_scheduling=False,
    enable_tiling=False,
)

MAX_PERFORMANCE_CONFIG = CompilerConfig(
    pass_config=PassConfig(opt_level=3),
    enable_scheduling=True,
    enable_tiling=True,
)

