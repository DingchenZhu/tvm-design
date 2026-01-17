"""
Operator fusion pass implementation.

Applies pattern-based fusion to combine multiple operators into
fused kernels for better performance.
"""

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.dataflow_pattern import *
from typing import List, Optional, Dict
import logging

from .base_pass import RelayPass, visualize_pass_effect
from .fusion_patterns import get_fusion_patterns, FusionPattern

logger = logging.getLogger(__name__)


class FusionPass(RelayPass):
    """
    Apply operator fusion using TVM's built-in fusion.
    
    TVM provides automatic fusion with different levels:
    - Level 0: No fusion
    - Level 1: Fuse element-wise and broadcast ops
    - Level 2: Level 1 + injective ops (reshape, transpose, etc.)
    - Level 3: Level 2 + Conv2D-like ops with element-wise epilogue
    - Level 4: Level 3 + all possible fusions
    """
    
    def __init__(self, fuse_opt_level: int = 3):
        """
        Initialize fusion pass.
        
        Args:
            fuse_opt_level: Fusion optimization level (0-4)
        """
        super().__init__("FuseOps", opt_level=2)
        self.fuse_opt_level = fuse_opt_level
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply operator fusion."""
        logger.info(f"Applying operator fusion (level={self.fuse_opt_level})")
        
        with tvm.transform.PassContext(opt_level=self.opt_level):
            pass_obj = relay.transform.FuseOps(fuse_opt_level=self.fuse_opt_level)
            mod = pass_obj(mod)
        
        return mod


class PatternBasedFusionCallback(DFPatternCallback):
    """
    Callback for pattern-based fusion.
    
    This allows custom fusion logic beyond what TVM provides automatically.
    """
    
    def __init__(self, pattern_name: str, pattern: DFPattern):
        """
        Initialize callback.
        
        Args:
            pattern_name: Name of the fusion pattern
            pattern: Dataflow pattern to match
        """
        super().__init__()
        self.pattern_name = pattern_name
        self.pattern = pattern
        self.require_type = True
        self.rewrite_once = False
    
    def callback(self, pre, post, node_map):
        """
        Callback invoked when pattern matches.
        
        Args:
            pre: Pattern before matching
            post: Matched pattern
            node_map: Mapping from pattern variables to matched expressions
            
        Returns:
            Transformed expression (or original if no transformation)
        """
        logger.debug(f"Matched pattern: {self.pattern_name}")
        
        # For most cases, TVM's automatic fusion handles the actual fusion
        # This callback is mainly for custom transformations or annotations
        
        # Return the original expression - TVM's FuseOps will handle the actual fusion
        return post


class CustomFusionPass(RelayPass):
    """
    Apply custom pattern-based fusion.
    
    This complements TVM's built-in fusion with custom patterns
    specific to VIS VPU hardware.
    """
    
    def __init__(
        self,
        patterns: Optional[List[FusionPattern]] = None,
        enable_tvm_fusion: bool = True,
        fuse_opt_level: int = 3
    ):
        """
        Initialize custom fusion pass.
        
        Args:
            patterns: List of fusion patterns to apply
                     If None, uses default patterns
            enable_tvm_fusion: Whether to also apply TVM's built-in fusion
            fuse_opt_level: TVM fusion optimization level
        """
        super().__init__("CustomFusion", opt_level=2)
        
        if patterns is None:
            self.patterns = get_fusion_patterns()
        else:
            self.patterns = patterns
        
        self.enable_tvm_fusion = enable_tvm_fusion
        self.fuse_opt_level = fuse_opt_level
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply custom fusion patterns."""
        logger.info(f"Applying custom fusion ({len(self.patterns)} patterns)")
        
        # Apply pattern-based rewrites
        for pattern_obj in self.patterns:
            logger.debug(f"Trying pattern: {pattern_obj.name}")
            
            try:
                # Create callback for this pattern
                callback = PatternBasedFusionCallback(
                    pattern_obj.name,
                    pattern_obj.pattern
                )
                
                # Apply pattern rewriting
                func = mod['main']
                new_func = rewrite(callback, func)
                
                if new_func != func:
                    logger.info(f"  ✓ Applied pattern: {pattern_obj.name}")
                    mod = tvm.IRModule.from_expr(new_func)
                
            except Exception as e:
                logger.warning(f"  ✗ Pattern {pattern_obj.name} failed: {e}")
        
        # Apply TVM's built-in fusion
        if self.enable_tvm_fusion:
            logger.info("Applying TVM built-in fusion")
            fusion_pass = FusionPass(fuse_opt_level=self.fuse_opt_level)
            mod = fusion_pass(mod)
        
        return mod


class VISVPUFusionConstraints:
    """
    Hardware-specific fusion constraints for VIS VPU.
    
    Defines which operations can be fused together based on
    hardware capabilities.
    """
    
    # Operations that can be fused as conv2d epilogue
    CONV_EPILOGUE_OPS = [
        'nn.relu',
        'nn.prelu',
        'nn.leaky_relu',
        'add',
        'nn.bias_add',
        'clip',
    ]
    
    # Operations that can be fused elementwise
    ELEMENTWISE_FUSABLE = [
        'add',
        'subtract',
        'multiply',
        'divide',
        'sigmoid',
        'tanh',
        'relu',
        'exp',
        'log',
        'sqrt',
        'negative',
        'abs',
    ]
    
    # Maximum number of ops in a fused kernel
    MAX_FUSED_OPS = 5
    
    @classmethod
    def can_fuse(cls, op1: str, op2: str) -> bool:
        """
        Check if two operations can be fused.
        
        Args:
            op1: First operation name
            op2: Second operation name
            
        Returns:
            True if operations can be fused
        """
        # Conv + epilogue
        if 'conv2d' in op1 and op2 in cls.CONV_EPILOGUE_OPS:
            return True
        
        # Elementwise fusion
        if op1 in cls.ELEMENTWISE_FUSABLE and op2 in cls.ELEMENTWISE_FUSABLE:
            return True
        
        # Dense + activation
        if 'dense' in op1 and op2 in cls.CONV_EPILOGUE_OPS:
            return True
        
        return False


def create_hardware_aware_fusion_pass(
    target: str = 'vis_vpu',
    opt_level: int = 3
) -> CustomFusionPass:
    """
    Create hardware-aware fusion pass for VIS VPU.
    
    Args:
        target: Target hardware ('vis_vpu', 'llvm', etc.)
        opt_level: Optimization level
        
    Returns:
        Configured fusion pass
    """
    if target == 'vis_vpu':
        # Use patterns optimized for VIS VPU
        patterns = get_fusion_patterns(include=[
            'conv2d_bias_relu',
            'conv2d_bias_prelu',
            'conv2d_bias',
            'dense_bias_relu',
            'dense_bias',
            'add_relu',
            'pixel_shuffle',
        ])
    else:
        # Use all patterns for generic targets
        patterns = get_fusion_patterns()
    
    return CustomFusionPass(
        patterns=patterns,
        enable_tvm_fusion=True,
        fuse_opt_level=opt_level
    )


def apply_fusion(
    mod: tvm.IRModule,
    fuse_opt_level: int = 3,
    custom_patterns: bool = True,
    target: str = 'vis_vpu'
) -> tvm.IRModule:
    """
    Convenience function to apply fusion optimizations.
    
    Args:
        mod: Input IR module
        fuse_opt_level: Fusion optimization level
        custom_patterns: Whether to apply custom patterns
        target: Target hardware
        
    Returns:
        Optimized IR module
    """
    if custom_patterns:
        fusion_pass = create_hardware_aware_fusion_pass(
            target=target,
            opt_level=fuse_opt_level
        )
    else:
        fusion_pass = FusionPass(fuse_opt_level=fuse_opt_level)
    
    return fusion_pass(mod)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing fusion pass...")
    
    try:
        from frontend import load_model
        from .base_pass import count_ops
        
        # Load model
        print("\n" + "="*60)
        print("Loading model...")
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Count ops before
        ops_before = count_ops(mod)
        print(f"\nOps before fusion: {sum(ops_before.values())}")
        
        # Apply fusion
        print("\n" + "="*60)
        print("Applying operator fusion...")
        mod_fused = apply_fusion(mod, fuse_opt_level=3, custom_patterns=True)
        
        # Count ops after
        ops_after = count_ops(mod_fused)
        print(f"\nOps after fusion: {sum(ops_after.values())}")
        
        # Visualize effect
        visualize_pass_effect(mod, mod_fused, "Operator Fusion")
        
        print("\n✓ Fusion pass test complete!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

