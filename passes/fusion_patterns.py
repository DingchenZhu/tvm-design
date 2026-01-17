"""
Operator fusion pattern definitions for Relay IR.

Defines patterns for fusing multiple operators into single compute kernels
to reduce memory traffic and improve performance.
"""

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import *
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FusionPattern:
    """Base class for fusion patterns."""
    
    def __init__(self, name: str, pattern: DFPattern, priority: int = 10):
        """
        Initialize fusion pattern.
        
        Args:
            name: Pattern name
            pattern: Dataflow pattern to match
            priority: Pattern priority (higher = applied first)
        """
        self.name = name
        self.pattern = pattern
        self.priority = priority
    
    def __repr__(self):
        return f"FusionPattern(name='{self.name}', priority={self.priority})"


def make_conv2d_bias_pattern(with_bias: bool = True) -> DFPattern:
    """
    Create pattern for Conv2D + Bias.
    
    Pattern: Conv2D + Bias (broadcast add)
    
    Args:
        with_bias: Whether to include bias addition
        
    Returns:
        Dataflow pattern
    """
    data = wildcard()
    weight = wildcard()
    conv = is_op('nn.conv2d')(data, weight)
    
    if with_bias:
        bias = wildcard()
        conv_bias = is_op('add')(conv, bias) | is_op('nn.bias_add')(conv, bias)
        return conv_bias
    
    return conv


def make_conv2d_bias_relu_pattern() -> DFPattern:
    """
    Create pattern for Conv2D + Bias + ReLU.
    
    This is one of the most common patterns in CNNs.
    """
    conv_bias = make_conv2d_bias_pattern(with_bias=True)
    relu = is_op('nn.relu')(conv_bias)
    return relu


def make_conv2d_bias_prelu_pattern() -> DFPattern:
    """
    Create pattern for Conv2D + Bias + PReLU.
    
    Common in super-resolution networks like FSRCNN.
    """
    conv_bias = make_conv2d_bias_pattern(with_bias=True)
    alpha = wildcard()
    prelu = is_op('nn.prelu')(conv_bias, alpha) | is_op('nn.leaky_relu')(conv_bias)
    return prelu


def make_conv2d_bn_pattern() -> DFPattern:
    """
    Create pattern for Conv2D + BatchNorm.
    
    Note: SimplifyInference should convert BN to scale+shift first.
    """
    data = wildcard()
    weight = wildcard()
    conv = is_op('nn.conv2d')(data, weight)
    
    gamma = wildcard()
    beta = wildcard()
    moving_mean = wildcard()
    moving_var = wildcard()
    
    bn = is_op('nn.batch_norm')(conv, gamma, beta, moving_mean, moving_var)
    # BatchNorm returns a tuple, get the first element
    bn_out = is_tuple_get_item(bn, 0)
    
    return bn_out


def make_conv2d_bn_relu_pattern() -> DFPattern:
    """
    Create pattern for Conv2D + BatchNorm + ReLU.
    """
    bn_out = make_conv2d_bn_pattern()
    relu = is_op('nn.relu')(bn_out)
    return relu


def make_dense_bias_pattern(with_bias: bool = True) -> DFPattern:
    """
    Create pattern for Dense (fully connected) + Bias.
    """
    data = wildcard()
    weight = wildcard()
    dense = is_op('nn.dense')(data, weight)
    
    if with_bias:
        bias = wildcard()
        dense_bias = is_op('add')(dense, bias) | is_op('nn.bias_add')(dense, bias)
        return dense_bias
    
    return dense


def make_dense_bias_relu_pattern() -> DFPattern:
    """
    Create pattern for Dense + Bias + ReLU.
    """
    dense_bias = make_dense_bias_pattern(with_bias=True)
    relu = is_op('nn.relu')(dense_bias)
    return relu


def make_add_relu_pattern() -> DFPattern:
    """
    Create pattern for Add + ReLU.
    
    Common in residual connections.
    """
    lhs = wildcard()
    rhs = wildcard()
    add = is_op('add')(lhs, rhs)
    relu = is_op('nn.relu')(add)
    return relu


def make_mul_add_pattern() -> DFPattern:
    """
    Create pattern for Multiply + Add.
    
    Can represent affine transformations.
    """
    data = wildcard()
    scale = wildcard()
    bias = wildcard()
    
    mul = is_op('multiply')(data, scale)
    add = is_op('add')(mul, bias)
    
    return add


def make_pixel_shuffle_pattern() -> DFPattern:
    """
    Create pattern for PixelShuffle operation.
    
    PixelShuffle is typically: Reshape → Transpose → Reshape
    Used for upsampling in super-resolution.
    """
    data = wildcard()
    
    # Pattern: reshape → transpose → reshape
    reshape1 = is_op('reshape')(data)
    transpose = is_op('transpose')(reshape1)
    reshape2 = is_op('reshape')(transpose)
    
    return reshape2


def make_global_avg_pool_pattern() -> DFPattern:
    """
    Create pattern for GlobalAvgPool.
    
    Can be implemented as: mean(axis=[2,3]) or adaptive_avg_pool2d
    """
    data = wildcard()
    
    # Pattern 1: mean over spatial dimensions
    mean_pattern = is_op('mean')(data)
    
    # Pattern 2: adaptive_avg_pool2d with output_size=1
    adaptive_pool_pattern = is_op('nn.adaptive_avg_pool2d')(data)
    
    # Pattern 3: global_avg_pool2d
    global_pool_pattern = is_op('nn.global_avg_pool2d')(data)
    
    return mean_pattern | adaptive_pool_pattern | global_pool_pattern


def make_silu_pattern() -> DFPattern:
    """
    Create pattern for SiLU (Swish) activation.
    
    SiLU(x) = x * sigmoid(x)
    """
    data = wildcard()
    sigmoid = is_op('sigmoid')(data)
    mul = is_op('multiply')(data, sigmoid)
    return mul


def make_gelu_pattern() -> DFPattern:
    """
    Create pattern for GELU activation.
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    Or simpler: x * sigmoid(1.702 * x)
    """
    data = wildcard()
    
    # Simplified GELU pattern
    # x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2 = is_constant()
    div = is_op('divide')(data, sqrt_2)
    erf = is_op('erf')(div)
    one = is_constant()
    add_one = is_op('add')(erf, one)
    mul1 = is_op('multiply')(data, add_one)
    half = is_constant()
    mul2 = is_op('multiply')(mul1, half)
    
    return mul2


# Pre-defined fusion patterns with priorities
FUSION_PATTERNS = [
    # Conv patterns (high priority)
    FusionPattern("conv2d_bias_relu", make_conv2d_bias_relu_pattern(), priority=100),
    FusionPattern("conv2d_bias_prelu", make_conv2d_bias_prelu_pattern(), priority=100),
    FusionPattern("conv2d_bn_relu", make_conv2d_bn_relu_pattern(), priority=95),
    FusionPattern("conv2d_bn", make_conv2d_bn_pattern(), priority=90),
    FusionPattern("conv2d_bias", make_conv2d_bias_pattern(with_bias=True), priority=85),
    
    # Dense patterns
    FusionPattern("dense_bias_relu", make_dense_bias_relu_pattern(), priority=90),
    FusionPattern("dense_bias", make_dense_bias_pattern(with_bias=True), priority=85),
    
    # Activation patterns
    FusionPattern("add_relu", make_add_relu_pattern(), priority=80),
    FusionPattern("silu", make_silu_pattern(), priority=80),
    FusionPattern("gelu", make_gelu_pattern(), priority=80),
    
    # Other patterns
    FusionPattern("mul_add", make_mul_add_pattern(), priority=70),
    FusionPattern("pixel_shuffle", make_pixel_shuffle_pattern(), priority=75),
    FusionPattern("global_avg_pool", make_global_avg_pool_pattern(), priority=70),
]


def get_fusion_patterns(include: Optional[List[str]] = None) -> List[FusionPattern]:
    """
    Get list of fusion patterns.
    
    Args:
        include: Optional list of pattern names to include
                If None, returns all patterns
    
    Returns:
        List of fusion patterns, sorted by priority
    """
    if include is None:
        patterns = FUSION_PATTERNS
    else:
        patterns = [p for p in FUSION_PATTERNS if p.name in include]
    
    # Sort by priority (highest first)
    patterns = sorted(patterns, key=lambda p: p.priority, reverse=True)
    
    logger.info(f"Using {len(patterns)} fusion patterns")
    for pattern in patterns:
        logger.debug(f"  - {pattern.name} (priority={pattern.priority})")
    
    return patterns


def print_fusion_patterns():
    """Print all available fusion patterns."""
    print("\nAvailable Fusion Patterns:")
    print("=" * 60)
    for pattern in sorted(FUSION_PATTERNS, key=lambda p: p.priority, reverse=True):
        print(f"  {pattern.name:30s} priority={pattern.priority}")
    print("=" * 60)
    print(f"Total: {len(FUSION_PATTERNS)} patterns")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing fusion pattern definitions...")
    
    # Print all patterns
    print_fusion_patterns()
    
    # Test pattern retrieval
    print("\n" + "="*60)
    print("Testing pattern retrieval:")
    
    all_patterns = get_fusion_patterns()
    print(f"All patterns: {len(all_patterns)}")
    
    conv_patterns = get_fusion_patterns(include=['conv2d_bias_relu', 'conv2d_bn'])
    print(f"Conv patterns: {len(conv_patterns)}")
    for p in conv_patterns:
        print(f"  - {p.name}")
    
    print("\n✓ Fusion pattern test complete!")

