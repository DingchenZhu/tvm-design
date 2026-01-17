"""
Base classes and utilities for implementing Relay optimization passes.
"""

import tvm
from tvm import relay
from tvm.relay import transform
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RelayPass:
    """Base class for Relay optimization passes."""
    
    def __init__(self, name: str, opt_level: int = 2):
        """
        Initialize pass.
        
        Args:
            name: Pass name
            opt_level: Optimization level (0-3)
        """
        self.name = name
        self.opt_level = opt_level
        self.enabled = True
    
    def __call__(self, mod: tvm.IRModule) -> tvm.IRModule:
        """
        Apply pass to module.
        
        Args:
            mod: Input IR module
            
        Returns:
            Transformed IR module
        """
        if not self.enabled:
            logger.info(f"Pass {self.name} is disabled, skipping")
            return mod
        
        logger.info(f"Applying pass: {self.name}")
        try:
            mod = self.transform(mod)
            logger.info(f"Pass {self.name} completed successfully")
            return mod
        except Exception as e:
            logger.error(f"Pass {self.name} failed: {e}")
            raise
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """
        Implement the actual transformation.
        
        Args:
            mod: Input IR module
            
        Returns:
            Transformed IR module
        """
        raise NotImplementedError("Subclasses must implement transform()")
    
    def enable(self):
        """Enable this pass."""
        self.enabled = True
    
    def disable(self):
        """Disable this pass."""
        self.enabled = False


class TVMBuiltinPass(RelayPass):
    """Wrapper for TVM built-in passes."""
    
    def __init__(self, name: str, pass_func, opt_level: int = 2, **kwargs):
        """
        Initialize TVM built-in pass wrapper.
        
        Args:
            name: Pass name
            pass_func: TVM pass function
            opt_level: Optimization level
            **kwargs: Arguments to pass to the pass function
        """
        super().__init__(name, opt_level)
        self.pass_func = pass_func
        self.pass_kwargs = kwargs
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply TVM built-in pass."""
        with tvm.transform.PassContext(opt_level=self.opt_level):
            pass_obj = self.pass_func(**self.pass_kwargs)
            mod = pass_obj(mod)
        return mod


class SequentialPass(RelayPass):
    """Apply multiple passes sequentially."""
    
    def __init__(self, name: str, passes: List[RelayPass], opt_level: int = 2):
        """
        Initialize sequential pass.
        
        Args:
            name: Pass name
            passes: List of passes to apply
            opt_level: Optimization level
        """
        super().__init__(name, opt_level)
        self.passes = passes
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply all passes sequentially."""
        for pass_obj in self.passes:
            if pass_obj.enabled:
                mod = pass_obj(mod)
        return mod
    
    def add_pass(self, pass_obj: RelayPass):
        """Add a pass to the sequence."""
        self.passes.append(pass_obj)


class ConditionalPass(RelayPass):
    """Apply pass conditionally based on a predicate."""
    
    def __init__(
        self,
        name: str,
        pass_obj: RelayPass,
        condition_func,
        opt_level: int = 2
    ):
        """
        Initialize conditional pass.
        
        Args:
            name: Pass name
            pass_obj: Pass to apply
            condition_func: Function that returns True if pass should be applied
            opt_level: Optimization level
        """
        super().__init__(name, opt_level)
        self.pass_obj = pass_obj
        self.condition_func = condition_func
    
    def transform(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply pass if condition is met."""
        if self.condition_func(mod):
            logger.info(f"Condition met for {self.name}, applying pass")
            mod = self.pass_obj(mod)
        else:
            logger.info(f"Condition not met for {self.name}, skipping")
        return mod


def count_ops(mod: tvm.IRModule) -> Dict[str, int]:
    """
    Count operators in the module.
    
    Args:
        mod: IR module
        
    Returns:
        Dictionary mapping operator names to counts
    """
    from collections import Counter
    
    class OpCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.ops = []
        
        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.ops.append(call.op.name)
            super().visit_call(call)
    
    counter = OpCounter()
    counter.visit(mod['main'])
    return dict(Counter(counter.ops))


def get_num_params(params: Dict[str, tvm.nd.NDArray]) -> int:
    """
    Get total number of parameters.
    
    Args:
        params: Parameters dictionary
        
    Returns:
        Total parameter count
    """
    import numpy as np
    total = 0
    for param in params.values():
        total += int(np.prod(param.shape))
    return total


def visualize_pass_effect(
    mod_before: tvm.IRModule,
    mod_after: tvm.IRModule,
    pass_name: str
):
    """
    Visualize the effect of a pass.
    
    Args:
        mod_before: Module before pass
        mod_after: Module after pass
        pass_name: Name of the pass
    """
    ops_before = count_ops(mod_before)
    ops_after = count_ops(mod_after)
    
    total_before = sum(ops_before.values())
    total_after = sum(ops_after.values())
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Pass: {pass_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total ops before: {total_before}")
    logger.info(f"Total ops after:  {total_after}")
    logger.info(f"Reduction:        {total_before - total_after} ops ({(1 - total_after/max(total_before, 1))*100:.1f}%)")
    
    # Show operator changes
    all_ops = set(ops_before.keys()) | set(ops_after.keys())
    changed = False
    for op in sorted(all_ops):
        before = ops_before.get(op, 0)
        after = ops_after.get(op, 0)
        if before != after:
            if not changed:
                logger.info(f"\nOperator changes:")
                changed = True
            logger.info(f"  {op}: {before} → {after}")
    
    if not changed:
        logger.info("No operator count changes")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test pass infrastructure
    print("Testing pass infrastructure...")
    
    # Create dummy passes
    pass1 = TVMBuiltinPass("InferType", relay.transform.InferType, opt_level=0)
    pass2 = TVMBuiltinPass("FoldConstant", relay.transform.FoldConstant)
    
    # Create sequential pass
    seq_pass = SequentialPass("BasicOptimization", [pass1, pass2])
    
    print(f"Created sequential pass with {len(seq_pass.passes)} passes")
    print("Pass infrastructure test complete!")

