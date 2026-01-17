"""
VIS VPU-specific fusion rules and constraints.

Defines hardware capabilities and fusion legality checks.
"""

from typing import List, Set, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class VISVPUFusionRules:
    """
    Fusion rules specific to VIS VPU hardware architecture.
    
    Based on the hardware capabilities defined in instruction.py and
    the existing VIS VPU backend.
    """
    
    # Operators supported natively by VIS VPU
    NATIVE_OPS = {
        'nn.conv2d',
        'nn.conv2d_transpose',
        'nn.dense',
        'nn.relu',
        'nn.prelu',
        'nn.leaky_relu',
        'nn.avg_pool2d',
        'nn.max_pool2d',
        'nn.global_avg_pool2d',
        'add',
        'multiply',
        'subtract',
        'divide',
        'sigmoid',
        'tanh',
        'clip',
        'reshape',
        'transpose',
        'concatenate',
    }
    
    # Operators that can be fused into conv2d
    CONV_FUSABLE_EPILOGUE = {
        'add',           # Bias add
        'nn.bias_add',   # Bias add (specialized)
        'nn.relu',       # ReLU activation
        'nn.prelu',      # PReLU activation
        'nn.leaky_relu', # Leaky ReLU
        'clip',          # Clip (for ReLU6)
        'multiply',      # Scale
    }
    
    # Maximum number of operations in a fused kernel
    MAX_FUSION_DEPTH = 4
    
    # Register file constraints
    MAX_ACCUMULATOR_REGS = 2  # From instruction.py
    MAX_LINE_BUFFERS = 2      # Ping-pong buffers
    MAX_WEIGHT_BUFFERS = 3    # From instruction.py
    
    @classmethod
    def can_fuse_conv_epilogue(cls, conv_op: str, epilogue_ops: List[str]) -> bool:
        """
        Check if epilogue operations can be fused into convolution.
        
        Args:
            conv_op: Convolution operation ('nn.conv2d', 'nn.dense', etc.)
            epilogue_ops: List of operations to fuse after conv
            
        Returns:
            True if fusion is legal
        """
        # Check conv is supported
        if conv_op not in ['nn.conv2d', 'nn.dense', 'nn.conv2d_transpose']:
            return False
        
        # Check depth limit
        if len(epilogue_ops) > cls.MAX_FUSION_DEPTH:
            logger.warning(f"Fusion depth {len(epilogue_ops)} exceeds limit {cls.MAX_FUSION_DEPTH}")
            return False
        
        # Check all epilogue ops are fusable
        for op in epilogue_ops:
            if op not in cls.CONV_FUSABLE_EPILOGUE:
                logger.debug(f"Op {op} cannot be fused into conv epilogue")
                return False
        
        # Specific patterns that are supported
        # Pattern 1: Conv + Bias + Activation
        if len(epilogue_ops) <= 2:
            return True
        
        # Pattern 2: Conv + Bias + Scale + Activation
        if len(epilogue_ops) == 3:
            # Check for reasonable ordering
            if epilogue_ops[0] in ['add', 'nn.bias_add']:
                return True
        
        return True
    
    @classmethod
    def can_fuse_elementwise(cls, ops: List[str]) -> bool:
        """
        Check if elementwise operations can be fused.
        
        Args:
            ops: List of elementwise operations
            
        Returns:
            True if fusion is legal
        """
        elementwise_ops = {
            'add', 'subtract', 'multiply', 'divide',
            'sigmoid', 'tanh', 'exp', 'log', 'sqrt',
            'negative', 'abs', 'relu', 'clip'
        }
        
        # Check all ops are elementwise
        if not all(op in elementwise_ops for op in ops):
            return False
        
        # Check depth limit
        if len(ops) > cls.MAX_FUSION_DEPTH:
            return False
        
        return True
    
    @classmethod
    def get_fusion_benefit(cls, op1: str, op2: str) -> float:
        """
        Estimate performance benefit of fusing two operations.
        
        Args:
            op1: First operation
            op2: Second operation
            
        Returns:
            Benefit score (higher = more beneficial)
        """
        # Conv + activation: high benefit (saves memory bandwidth)
        if 'conv2d' in op1 and op2 in cls.CONV_FUSABLE_EPILOGUE:
            return 10.0
        
        # Dense + activation: high benefit
        if 'dense' in op1 and op2 in cls.CONV_FUSABLE_EPILOGUE:
            return 10.0
        
        # Elementwise fusion: medium benefit
        if op1 in ['add', 'multiply'] and op2 in ['add', 'multiply']:
            return 5.0
        
        # Activation fusion: medium benefit
        if op1 in ['sigmoid', 'tanh'] and op2 in ['multiply', 'add']:
            return 5.0
        
        # Layout ops: low benefit (may be necessary for correctness)
        if op1 in ['reshape', 'transpose'] or op2 in ['reshape', 'transpose']:
            return 2.0
        
        # Default: minimal benefit
        return 1.0
    
    @classmethod
    def suggest_fusion_strategy(cls, op_sequence: List[str]) -> Dict:
        """
        Suggest optimal fusion strategy for a sequence of operations.
        
        Args:
            op_sequence: List of operations in order
            
        Returns:
            Dictionary with fusion suggestions
        """
        suggestions = {
            'fusable': [],
            'splits': [],
            'reasons': []
        }
        
        current_group = []
        
        for i, op in enumerate(op_sequence):
            if not current_group:
                current_group = [op]
                continue
            
            # Check if can add to current group
            base_op = current_group[0]
            epilogue = current_group[1:] + [op]
            
            if 'conv2d' in base_op or 'dense' in base_op:
                if cls.can_fuse_conv_epilogue(base_op, epilogue):
                    current_group.append(op)
                else:
                    # Start new group
                    suggestions['fusable'].append(current_group)
                    suggestions['splits'].append(i)
                    current_group = [op]
            elif cls.can_fuse_elementwise(current_group + [op]):
                current_group.append(op)
            else:
                # Cannot fuse, start new group
                suggestions['fusable'].append(current_group)
                suggestions['splits'].append(i)
                current_group = [op]
        
        # Add last group
        if current_group:
            suggestions['fusable'].append(current_group)
        
        return suggestions


class FusionCostModel:
    """
    Cost model for evaluating fusion decisions.
    
    Estimates performance impact of different fusion strategies.
    """
    
    # Relative costs (in arbitrary units)
    MEMORY_BANDWIDTH_COST = 10.0  # Cost per memory transfer
    COMPUTE_COST = 1.0            # Cost per compute op
    REGISTER_SPILL_COST = 5.0     # Cost of register spilling
    
    @classmethod
    def estimate_cost(cls, ops: List[str], fused: bool = False) -> float:
        """
        Estimate execution cost.
        
        Args:
            ops: List of operations
            fused: Whether operations are fused
            
        Returns:
            Estimated cost
        """
        if not ops:
            return 0.0
        
        # Compute cost
        compute_cost = len(ops) * cls.COMPUTE_COST
        
        if fused:
            # Fused: only load once, store once
            memory_cost = 2 * cls.MEMORY_BANDWIDTH_COST
        else:
            # Unfused: load and store for each op
            memory_cost = len(ops) * 2 * cls.MEMORY_BANDWIDTH_COST
        
        total_cost = compute_cost + memory_cost
        
        return total_cost
    
    @classmethod
    def fusion_benefit(cls, ops: List[str]) -> float:
        """
        Calculate benefit of fusing operations.
        
        Args:
            ops: List of operations to fuse
            
        Returns:
            Benefit (positive = fusion is beneficial)
        """
        unfused_cost = cls.estimate_cost(ops, fused=False)
        fused_cost = cls.estimate_cost(ops, fused=True)
        
        benefit = unfused_cost - fused_cost
        return benefit


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VIS VPU fusion rules...")
    
    rules = VISVPUFusionRules()
    
    # Test conv epilogue fusion
    print("\n" + "="*60)
    print("Testing Conv epilogue fusion:")
    
    test_cases = [
        ('nn.conv2d', ['add', 'nn.relu']),
        ('nn.conv2d', ['nn.bias_add', 'nn.prelu']),
        ('nn.conv2d', ['add', 'multiply', 'nn.relu']),
        ('nn.dense', ['add', 'sigmoid']),
    ]
    
    for conv_op, epilogue in test_cases:
        can_fuse = rules.can_fuse_conv_epilogue(conv_op, epilogue)
        benefit = rules.get_fusion_benefit(conv_op, epilogue[0])
        print(f"  {conv_op} + {epilogue}: {'✓' if can_fuse else '✗'} (benefit={benefit:.1f})")
    
    # Test fusion strategy
    print("\n" + "="*60)
    print("Testing fusion strategy:")
    
    op_seq = ['nn.conv2d', 'add', 'nn.relu', 'nn.max_pool2d', 'reshape']
    strategy = rules.suggest_fusion_strategy(op_seq)
    
    print(f"  Input sequence: {op_seq}")
    print(f"  Suggested groups: {strategy['fusable']}")
    print(f"  Split points: {strategy['splits']}")
    
    # Test cost model
    print("\n" + "="*60)
    print("Testing cost model:")
    
    cost_model = FusionCostModel()
    ops = ['nn.conv2d', 'add', 'nn.relu']
    
    unfused = cost_model.estimate_cost(ops, fused=False)
    fused = cost_model.estimate_cost(ops, fused=True)
    benefit = cost_model.fusion_benefit(ops)
    
    print(f"  Operations: {ops}")
    print(f"  Unfused cost: {unfused:.1f}")
    print(f"  Fused cost: {fused:.1f}")
    print(f"  Benefit: {benefit:.1f}")
    
    print("\n✓ Fusion rules test complete!")

