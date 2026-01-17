"""
Dead Code Elimination (DCE) pass implementation.

Removes unreachable and unused operations from the IR graph.
"""

import tvm
from tvm import relay
from tvm.relay import transform, ExprMutator, ExprVisitor
from typing import Set, Dict
import logging

from .base_pass import RelayPass, TVMBuiltinPass

logger = logging.getLogger(__name__)


class DeadCodeEliminationPass(TVMBuiltinPass):
    """
    Standard dead code elimination pass.
    
    Removes expressions that are not used in computing the output.
    """
    
    def __init__(self, inline_once: bool = False):
        """
        Initialize DCE pass.
        
        Args:
            inline_once: Whether to inline let bindings used once
        """
        super().__init__(
            name="DeadCodeElimination",
            pass_func=relay.transform.DeadCodeElimination,
            opt_level=2,
            inline_once=inline_once
        )


class LivenessAnalyzer(ExprVisitor):
    """
    Analyze which expressions are live (used).
    """
    
    def __init__(self):
        super().__init__()
        self.live_exprs: Set[relay.Expr] = set()
        self.var_uses: Dict[relay.Var, int] = {}
    
    def visit(self, expr):
        """Mark expression as live."""
        self.live_exprs.add(expr)
        super().visit(expr)
    
    def visit_var(self, var):
        """Track variable usage."""
        self.var_uses[var] = self.var_uses.get(var, 0) + 1
        super().visit_var(var)
    
    def visit_call(self, call):
        """Visit call expression."""
        # Mark call as live
        self.live_exprs.add(call)
        # Visit arguments
        for arg in call.args:
            self.visit(arg)
        super().visit_call(call)


class CustomDCE(ExprMutator):
    """
    Custom dead code elimination with additional optimizations.
    
    Removes:
    - Unused let bindings
    - Unused tuple elements
    - Debug/print operations
    - Identity operations (reshape to same shape, etc.)
    """
    
    def __init__(self):
        super().__init__()
        self.analyzer = LivenessAnalyzer()
    
    def visit_let(self, let):
        """Remove unused let bindings."""
        # Visit the body first
        new_body = self.visit(let.body)
        
        # Check if variable is used
        if let.var not in self.analyzer.var_uses or self.analyzer.var_uses[let.var] == 0:
            logger.debug(f"Removing unused let binding: {let.var.name_hint}")
            return new_body
        
        # Visit the value
        new_value = self.visit(let.value)
        
        if new_value == let.value and new_body == let.body:
            return let
        
        return relay.Let(let.var, new_value, new_body)
    
    def visit_tuple(self, tup):
        """Simplify tuples."""
        new_fields = [self.visit(field) for field in tup.fields]
        
        # If all fields are the same, return original
        if all(new_field == field for new_field, field in zip(new_fields, tup.fields)):
            return tup
        
        return relay.Tuple(new_fields)
    
    def visit_call(self, call):
        """Remove identity operations and debug ops."""
        # Check for identity operations
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name
            
            # Remove debug/print ops
            if 'debug' in op_name or 'print' in op_name:
                logger.debug(f"Removing debug op: {op_name}")
                # Return the input (assume first argument)
                if len(call.args) > 0:
                    return self.visit(call.args[0])
            
            # Identity reshape: reshape(x, x.shape)
            if op_name == 'reshape':
                # TODO: Check if reshape is actually changing the shape
                # For now, keep all reshapes as they might be needed for layout
                pass
            
            # Identity cast: cast(x, x.dtype)
            if op_name == 'cast':
                # TODO: Check if cast is actually changing the dtype
                pass
        
        # Visit arguments
        new_args = [self.visit(arg) for arg in call.args]
        
        if all(new_arg == arg for new_arg, arg in zip(new_args, call.args)):
            return call
        
        return relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)


class EnhancedDCEPass(RelayPass):
    """
    Enhanced dead code elimination with custom optimizations.
    """
    
    def __init__(self):
        super().__init__("EnhancedDCE", opt_level=2)
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply enhanced DCE."""
        logger.info("Applying enhanced dead code elimination")
        
        # First, apply standard DCE
        with tvm.transform.PassContext(opt_level=self.opt_level):
            std_dce = relay.transform.DeadCodeElimination()
            mod = std_dce(mod)
        
        # Then apply custom DCE
        try:
            mutator = CustomDCE()
            func = mod['main']
            
            # Analyze liveness first
            mutator.analyzer.visit(func)
            
            # Apply transformations
            new_func = mutator.visit(func)
            
            if new_func != func:
                logger.info("Custom DCE applied transformations")
                mod = tvm.IRModule.from_expr(new_func)
        except Exception as e:
            logger.warning(f"Custom DCE failed, using standard DCE only: {e}")
        
        return mod


def apply_dce(mod: tvm.IRModule, enhanced: bool = True) -> tvm.IRModule:
    """
    Convenience function to apply dead code elimination.
    
    Args:
        mod: Input IR module
        enhanced: Whether to use enhanced DCE with custom optimizations
        
    Returns:
        Optimized IR module
    """
    if enhanced:
        dce_pass = EnhancedDCEPass()
    else:
        dce_pass = DeadCodeEliminationPass()
    
    return dce_pass(mod)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing DCE pass...")
    
    try:
        from frontend import load_model
        from .base_pass import count_ops, visualize_pass_effect
        
        # Load model
        print("\n" + "="*60)
        print("Loading model...")
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Count ops before
        ops_before = count_ops(mod)
        print(f"\nOps before DCE: {sum(ops_before.values())}")
        
        # Apply DCE
        print("\n" + "="*60)
        print("Applying dead code elimination...")
        mod_dce = apply_dce(mod, enhanced=True)
        
        # Count ops after
        ops_after = count_ops(mod_dce)
        print(f"\nOps after DCE: {sum(ops_after.values())}")
        
        # Visualize effect
        visualize_pass_effect(mod, mod_dce, "Dead Code Elimination")
        
        print("\n✓ DCE pass test complete!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

