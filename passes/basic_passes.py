"""
Basic optimization passes for Relay IR.

Includes:
- Type inference
- Constant folding
- Expression simplification
- Dead code elimination (basic)
- Layout transformation
"""

import tvm
from tvm import relay
from tvm.relay import transform
from typing import Optional, Dict, Any
import logging

from .base_pass import RelayPass, TVMBuiltinPass, SequentialPass

logger = logging.getLogger(__name__)


class InferTypePass(TVMBuiltinPass):
    """
    Infer types for all expressions in the IR.
    
    This is a fundamental pass that must be run before many other optimizations.
    """
    
    def __init__(self):
        super().__init__(
            name="InferType",
            pass_func=relay.transform.InferType,
            opt_level=0
        )


class FoldConstantPass(TVMBuiltinPass):
    """
    Fold constant expressions at compile time.
    
    Examples:
    - 2 + 3 → 5
    - reshape(constant, shape) → reshaped constant
    - Broadcasting operations on constants
    """
    
    def __init__(self, fold_qnn: bool = False):
        """
        Initialize constant folding pass.
        
        Args:
            fold_qnn: Whether to fold QNN (quantized) operators
                     (only supported in TVM v0.11+, ignored in older versions)
        """
        # Check if fold_qnn parameter is supported by inspecting the function signature
        import inspect
        sig = inspect.signature(relay.transform.FoldConstant)
        kwargs = {}
        if 'fold_qnn' in sig.parameters:
            kwargs['fold_qnn'] = fold_qnn
        
        super().__init__(
            name="FoldConstant",
            pass_func=relay.transform.FoldConstant,
            opt_level=2,
            **kwargs
        )


class SimplifyInferencePass(TVMBuiltinPass):
    """
    Simplify operators specific to inference.
    
    Transformations:
    - BatchNorm → scale + shift (fuse running mean/var)
    - Dropout → identity (remove in inference mode)
    """
    
    def __init__(self):
        super().__init__(
            name="SimplifyInference",
            pass_func=relay.transform.SimplifyInference,
            opt_level=2
        )


class SimplifyExprPass(TVMBuiltinPass):
    """
    Apply algebraic simplifications.
    
    Examples:
    - x + 0 → x
    - x * 1 → x
    - x * 0 → 0
    - Eliminate identity operations
    """
    
    def __init__(self):
        super().__init__(
            name="SimplifyExpr",
            pass_func=relay.transform.SimplifyExpr,
            opt_level=2
        )


class FoldScaleAxisPass(TVMBuiltinPass):
    """
    Fold scale operations into conv/dense weights.
    
    Useful for optimizing batch normalization patterns:
    - Conv → BatchNorm → ... can fold scale into conv weights
    """
    
    def __init__(self):
        super().__init__(
            name="FoldScaleAxis",
            pass_func=relay.transform.FoldScaleAxis,
            opt_level=2
        )


class CanonicalizeCastPass(TVMBuiltinPass):
    """
    Canonicalize cast operations.
    
    - Remove redundant casts (cast(cast(x, dtype), dtype) → cast(x, dtype))
    - Simplify cast chains
    """
    
    def __init__(self):
        super().__init__(
            name="CanonicalizeCast",
            pass_func=relay.transform.CanonicalizeCast,
            opt_level=2
        )


class CanonicalizeOpsPass(TVMBuiltinPass):
    """
    Canonicalize operators to standard forms.
    
    - Rewrite operators to their canonical forms
    - Useful for pattern matching in later passes
    """
    
    def __init__(self):
        super().__init__(
            name="CanonicalizeOps",
            pass_func=relay.transform.CanonicalizeOps,
            opt_level=2
        )


class EliminateCommonSubexprPass(TVMBuiltinPass):
    """
    Eliminate common subexpressions.
    
    If the same expression is computed multiple times,
    compute it once and reuse the result.
    """
    
    def __init__(self, fskip: Optional[Any] = None):
        """
        Initialize CSE pass.
        
        Args:
            fskip: Optional function to skip certain expressions
        """
        kwargs = {}
        if fskip is not None:
            kwargs['fskip'] = fskip
        
        super().__init__(
            name="EliminateCommonSubexpr",
            pass_func=relay.transform.EliminateCommonSubexpr,
            opt_level=2,
            **kwargs
        )


class ConvertLayoutPass(RelayPass):
    """
    Convert data layout between different formats.
    
    Common transformations:
    - NCHW ↔ NHWC (batch, channel, height, width)
    - OIHW ↔ HWIO (weights)
    """
    
    def __init__(self, desired_layouts: Optional[Dict[str, list]] = None):
        """
        Initialize layout conversion pass.
        
        Args:
            desired_layouts: Dictionary mapping op names to desired layouts
                           e.g., {'nn.conv2d': ['NHWC', 'default']}
        """
        super().__init__("ConvertLayout", opt_level=2)
        self.desired_layouts = desired_layouts or {}
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply layout conversion."""
        if not self.desired_layouts:
            logger.info("No layout conversions specified, skipping")
            return mod
        
        with tvm.transform.PassContext(opt_level=self.opt_level):
            pass_obj = relay.transform.ConvertLayout(self.desired_layouts)
            mod = pass_obj(mod)
        
        return mod


class BasicOptimizationSequence(SequentialPass):
    """
    Standard sequence of basic optimization passes.
    
    This is the recommended starting point for most optimizations.
    """
    
    def __init__(
        self,
        opt_level: int = 2,
        simplify_inference: bool = True,
        fold_scale_axis: bool = True,
        eliminate_cse: bool = True
    ):
        """
        Initialize basic optimization sequence.
        
        Args:
            opt_level: Optimization level
            simplify_inference: Whether to simplify inference-specific ops
            fold_scale_axis: Whether to fold scale into weights
            eliminate_cse: Whether to eliminate common subexpressions
        """
        passes = [
            # Type inference first
            InferTypePass(),
            
            # Simplify inference-specific operations
            SimplifyInferencePass() if simplify_inference else None,
            
            # Fold constants
            FoldConstantPass(),
            
            # Simplify expressions
            SimplifyExprPass(),
            
            # Fold scale axis into weights
            FoldScaleAxisPass() if fold_scale_axis else None,
            
            # Canonicalize
            CanonicalizeCastPass(),
            CanonicalizeOpsPass(),
            
            # Eliminate common subexpressions
            EliminateCommonSubexprPass() if eliminate_cse else None,
            
            # Final constant fold and type inference
            FoldConstantPass(),
            InferTypePass(),
        ]
        
        # Filter out None passes
        passes = [p for p in passes if p is not None]
        
        super().__init__("BasicOptimization", passes, opt_level)


def apply_basic_optimizations(
    mod: tvm.IRModule,
    opt_level: int = 2,
    **kwargs
) -> tvm.IRModule:
    """
    Convenience function to apply basic optimizations.
    
    Args:
        mod: Input IR module
        opt_level: Optimization level (0-3)
        **kwargs: Additional arguments for BasicOptimizationSequence
        
    Returns:
        Optimized IR module
    """
    opt_seq = BasicOptimizationSequence(opt_level=opt_level, **kwargs)
    return opt_seq(mod)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing basic optimization passes...")
    
    # Test with a simple model
    try:
        from frontend import load_model
        
        # Load ONNX model
        print("\n" + "="*60)
        print("Loading ONNX model...")
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Count ops before
        from .base_pass import count_ops
        ops_before = count_ops(mod)
        print(f"\nOps before optimization: {sum(ops_before.values())}")
        
        # Apply optimizations
        print("\n" + "="*60)
        print("Applying basic optimizations...")
        mod_optimized = apply_basic_optimizations(mod, opt_level=2)
        
        # Count ops after
        ops_after = count_ops(mod_optimized)
        print(f"\nOps after optimization: {sum(ops_after.values())}")
        print(f"Reduction: {sum(ops_before.values()) - sum(ops_after.values())} ops")
        
        print("\n✓ Basic optimization passes test complete!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

