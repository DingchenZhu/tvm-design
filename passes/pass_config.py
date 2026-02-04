"""
Configuration for optimization passes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class PassConfig:
    """Configuration for pass pipeline."""
    
    # Optimization level (0-3)
    opt_level: int = 2
    
    # Target hardware
    target: str = 'vis_vpu'
    
    # Pass selection
    enable_fusion: bool = True
    enable_dce: bool = True
    enable_cse: bool = True
    enable_simplify: bool = True
    enable_fold_constant: bool = False  # Requires LLVM, disabled by default
    enable_fold_scale_axis: bool = False  # Requires LLVM, disabled by default
    
    # Fusion settings
    fusion_level: int = 3
    custom_fusion_patterns: bool = True
    
    # Layout optimization
    preferred_layout: Optional[str] = 'NHWC'  # or 'NCHW'
    
    # Debug settings
    visualize_passes: bool = False
    dump_ir: bool = False
    
    # Additional pass names to disable
    disabled_passes: List[str] = field(default_factory=list)
    
    # Additional pass names to enable
    enabled_passes: List[str] = field(default_factory=list)


# Preset configurations
VIS_VPU_CONFIG = PassConfig(
    opt_level=3,
    target='vis_vpu',
    enable_fusion=True,
    fusion_level=3,
    custom_fusion_patterns=True,
    preferred_layout='NHWC',
)

FAST_COMPILE_CONFIG = PassConfig(
    opt_level=1,
    enable_fusion=True,
    fusion_level=2,
    enable_cse=False,
)

MAX_PERFORMANCE_CONFIG = PassConfig(
    opt_level=3,
    enable_fusion=True,
    fusion_level=3,
    custom_fusion_patterns=True,
    enable_cse=True,
)

