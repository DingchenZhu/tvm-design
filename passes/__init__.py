"""
IR optimization passes for TVM Relay.

Includes:
- Operator fusion
- Dead code elimination
- Common subexpression elimination
- Redundancy elimination
- Quantization optimization
"""

from .base_pass import (
    RelayPass,
    TVMBuiltinPass,
    SequentialPass,
    ConditionalPass,
    count_ops,
    visualize_pass_effect,
)

from .basic_passes import (
    InferTypePass,
    FoldConstantPass,
    SimplifyInferencePass,
    SimplifyExprPass,
    FoldScaleAxisPass,
    CanonicalizeCastPass,
    CanonicalizeOpsPass,
    EliminateCommonSubexprPass,
    ConvertLayoutPass,
    BasicOptimizationSequence,
    apply_basic_optimizations,
)

from .fusion_pass import (
    FusionPass,
    CustomFusionPass,
    create_hardware_aware_fusion_pass,
    apply_fusion,
)

from .dce_pass import (
    DeadCodeEliminationPass,
    EnhancedDCEPass,
    apply_dce,
)

from .redundancy_elimination import (
    CSEPass,
    RedundancyEliminationPass,
    apply_redundancy_elimination,
)

from .pass_manager import (
    PassPipeline,
    PassManager,
    create_basic_pipeline,
    create_standard_pipeline,
    create_aggressive_pipeline,
    create_visvpu_pipeline,
)

from .pass_config import (
    PassConfig,
    VIS_VPU_CONFIG,
    FAST_COMPILE_CONFIG,
    MAX_PERFORMANCE_CONFIG,
)

__all__ = [
    # Base classes
    'RelayPass',
    'TVMBuiltinPass',
    'SequentialPass',
    'ConditionalPass',
    # Utilities
    'count_ops',
    'visualize_pass_effect',
    # Basic passes
    'InferTypePass',
    'FoldConstantPass',
    'SimplifyInferencePass',
    'SimplifyExprPass',
    'FoldScaleAxisPass',
    'CanonicalizeCastPass',
    'CanonicalizeOpsPass',
    'EliminateCommonSubexprPass',
    'ConvertLayoutPass',
    'BasicOptimizationSequence',
    'apply_basic_optimizations',
    # Fusion
    'FusionPass',
    'CustomFusionPass',
    'create_hardware_aware_fusion_pass',
    'apply_fusion',
    # DCE
    'DeadCodeEliminationPass',
    'EnhancedDCEPass',
    'apply_dce',
    # CSE and redundancy
    'CSEPass',
    'RedundancyEliminationPass',
    'apply_redundancy_elimination',
    # Pass management
    'PassPipeline',
    'PassManager',
    'create_basic_pipeline',
    'create_standard_pipeline',
    'create_aggressive_pipeline',
    'create_visvpu_pipeline',
    # Configuration
    'PassConfig',
    'VIS_VPU_CONFIG',
    'FAST_COMPILE_CONFIG',
    'MAX_PERFORMANCE_CONFIG',
]

