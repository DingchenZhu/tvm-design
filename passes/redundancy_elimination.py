"""
Common Subexpression Elimination (CSE) and redundancy elimination.

Identifies and eliminates:
- Common subexpressions (identical computations)
- Redundant operations (no-ops, identity operations)
- Duplicate constants
"""

import tvm
from tvm import relay
from tvm.relay import transform, ExprMutator, structural_hash
from typing import Dict, Any
import logging

from .base_pass import RelayPass, TVMBuiltinPass

logger = logging.getLogger(__name__)


class CSEPass(TVMBuiltinPass):
    """
    Common subexpression elimination using TVM's built-in pass.
    """
    
    def __init__(self):
        super().__init__(
            name="EliminateCommonSubexpr",
            pass_func=relay.transform.EliminateCommonSubexpr,
            opt_level=2
        )


class RedundancyEliminator(ExprMutator):
    """
    Eliminate redundant operations.
    
    Handles:
    - Identity operations (x + 0, x * 1, etc.)
    - No-op operations (reshape to same shape)
    - Duplicate computations
    """
    
    def __init__(self):
        super().__init__()
        self.expr_cache: Dict[int, relay.Expr] = {}
        self.eliminated = 0
    
    def visit_call(self, call):
        """Visit and potentially eliminate redundant calls."""
        # Check cache for identical expression
        expr_hash = structural_hash(call)
        
        if expr_hash in self.expr_cache:
            cached_expr = self.expr_cache[expr_hash]
            logger.debug(f"Reusing cached expression (hash={expr_hash})")
            self.eliminated += 1
            return cached_expr
        
        # Check for specific redundancy patterns
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name
            
            # Handle identity operations
            new_call = self._simplify_identity(call, op_name)
            if new_call != call:
                return new_call
        
        # Visit arguments
        new_args = [self.visit(arg) for arg in call.args]
        
        if all(new_arg == arg for new_arg, arg in zip(new_args, call.args)):
            result = call
        else:
            result = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
        
        # Cache the result
        self.expr_cache[expr_hash] = result
        
        return result
    
    def _simplify_identity(self, call, op_name):
        """Simplify identity operations."""
        args = call.args
        
        # x + 0 = x
        if op_name == 'add' and len(args) == 2:
            if self._is_zero_constant(args[1]):
                logger.debug("Eliminating x + 0")
                return self.visit(args[0])
            if self._is_zero_constant(args[0]):
                logger.debug("Eliminating 0 + x")
                return self.visit(args[1])
        
        # x * 1 = x
        if op_name == 'multiply' and len(args) == 2:
            if self._is_one_constant(args[1]):
                logger.debug("Eliminating x * 1")
                return self.visit(args[0])
            if self._is_one_constant(args[0]):
                logger.debug("Eliminating 1 * x")
                return self.visit(args[1])
        
        # x * 0 = 0
        if op_name == 'multiply' and len(args) == 2:
            if self._is_zero_constant(args[1]):
                logger.debug("Eliminating x * 0")
                return args[1]
            if self._is_zero_constant(args[0]):
                logger.debug("Eliminating 0 * x")
                return args[0]
        
        # x - 0 = x
        if op_name == 'subtract' and len(args) == 2:
            if self._is_zero_constant(args[1]):
                logger.debug("Eliminating x - 0")
                return self.visit(args[0])
        
        # x / 1 = x
        if op_name == 'divide' and len(args) == 2:
            if self._is_one_constant(args[1]):
                logger.debug("Eliminating x / 1")
                return self.visit(args[0])
        
        return call
    
    def _is_zero_constant(self, expr) -> bool:
        """Check if expression is constant zero."""
        if isinstance(expr, relay.Constant):
            data = expr.data.numpy()
            return (data == 0).all()
        return False
    
    def _is_one_constant(self, expr) -> bool:
        """Check if expression is constant one."""
        if isinstance(expr, relay.Constant):
            data = expr.data.numpy()
            return (data == 1).all()
        return False


class RedundancyEliminationPass(RelayPass):
    """
    Redundancy elimination pass combining CSE and custom optimizations.
    """
    
    def __init__(self, apply_cse: bool = True):
        """
        Initialize redundancy elimination pass.
        
        Args:
            apply_cse: Whether to also apply TVM's CSE pass
        """
        super().__init__("RedundancyElimination", opt_level=2)
        self.apply_cse = apply_cse
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply redundancy elimination."""
        logger.info("Applying redundancy elimination")
        
        # Apply TVM's CSE first
        if self.apply_cse:
            with tvm.transform.PassContext(opt_level=self.opt_level):
                cse_pass = relay.transform.EliminateCommonSubexpr()
                mod = cse_pass(mod)
        
        # Apply custom redundancy elimination
        try:
            eliminator = RedundancyEliminator()
            func = mod['main']
            
            new_func = eliminator.visit(func)
            
            if eliminator.eliminated > 0:
                logger.info(f"Eliminated {eliminator.eliminated} redundant expressions")
                mod = tvm.IRModule.from_expr(new_func)
        except Exception as e:
            logger.warning(f"Custom redundancy elimination failed: {e}")
        
        return mod


def apply_redundancy_elimination(
    mod: tvm.IRModule,
    apply_cse: bool = True
) -> tvm.IRModule:
    """
    Convenience function to apply redundancy elimination.
    
    Args:
        mod: Input IR module
        apply_cse: Whether to also apply CSE
        
    Returns:
        Optimized IR module
    """
    pass_obj = RedundancyEliminationPass(apply_cse=apply_cse)
    return pass_obj(mod)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing redundancy elimination...")
    
    try:
        from frontend import load_model
        from .base_pass import count_ops, visualize_pass_effect
        
        # Load model
        print("\n" + "="*60)
        print("Loading model...")
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Count ops before
        ops_before = count_ops(mod)
        print(f"\nOps before: {sum(ops_before.values())}")
        
        # Apply redundancy elimination
        print("\n" + "="*60)
        print("Applying redundancy elimination...")
        mod_optimized = apply_redundancy_elimination(mod, apply_cse=True)
        
        # Count ops after
        ops_after = count_ops(mod_optimized)
        print(f"\nOps after: {sum(ops_after.values())}")
        
        # Visualize effect
        visualize_pass_effect(mod, mod_optimized, "Redundancy Elimination")
        
        print("\n✓ Redundancy elimination test complete!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

