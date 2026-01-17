# TVM-Based Compiler Stack for VIS VPU

A complete compiler infrastructure that converts PyTorch/ONNX neural network models to VIS VPU hardware instructions using Apache TVM.

## Overview

This compiler stack provides an end-to-end solution for compiling deep learning models to VIS VPU hardware, with emphasis on:
- IR optimization passes (operator fusion, DCE, redundant node elimination)
- Custom scheduling and tiling strategies
- Memory allocation optimization
- Hardware-specific code generation

## Supported Models

- **FSRCNN (PyTorch)**: Super-resolution model with custom DeformableConv2d operators
- **USR_Net (ONNX)**: ONNX-based super-resolution network

## Architecture

```
Model (PyTorch/ONNX) → Relay IR → Optimized IR → TIR → VIS VPU Instructions
```

### Pipeline Stages

1. **Frontend**: Convert PyTorch/ONNX models to Relay IR
2. **Optimization Passes**: Apply IR transformations (fusion, DCE, CSE, etc.)
3. **Scheduling**: Tiling and memory layout optimization
4. **Codegen**: Generate VIS VPU instruction sequences
5. **Assembly**: Binary code generation

## Directory Structure

```
├── frontend/          # Model importers (PyTorch, ONNX)
├── passes/            # IR optimization passes
├── scheduling/        # Scheduling and tiling strategies
├── codegen/           # Code generation for VIS VPU
├── compiler/          # Compiler driver and configuration
├── tests/             # Test suites
└── examples/          # Example compilation scripts
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build TVM (if not already installed)
# See: https://tvm.apache.org/docs/install/from_source.html
```

## Usage

### Compile ONNX Model

```python
from compiler.driver import TVMVISCompiler
from compiler.config import CompilerConfig

# Configure compiler
config = CompilerConfig(
    opt_level=3,
    target='vis_vpu',
    enable_fusion=True
)

# Compile model
compiler = TVMVISCompiler(config)
binary = compiler.compile('USR_Net.onnx', model_type='onnx')
```

### Compile PyTorch Model

```python
from models_new_930 import FSRCNN
import torch

# Load model
model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
model.eval()

# Compile
example_input = torch.randn(1, 1, 270, 480)
binary = compiler.compile(model, model_type='pytorch', example_input=example_input)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_passes.py

# Run with coverage
pytest --cov=. tests/
```

### Code Formatting

```bash
black frontend/ passes/ scheduling/ codegen/ compiler/
```

## Key Features

### IR Optimization Passes

- **Operator Fusion**: Conv2D+BiasAdd+ReLU, Conv2D+PReLU patterns
- **Dead Code Elimination**: Remove unreachable and unused operations
- **Common Subexpression Elimination**: Eliminate redundant computations
- **Constant Folding**: Compute constant expressions at compile time
- **Layout Optimization**: Transform NCHW ↔ NHWC for hardware efficiency

### Custom Operator Support

- DeformableConv2d with offset generation
- PixelShuffle for upsampling
- PReLU activations

### Hardware-Specific Optimizations

- Line buffer ping-pong management
- Weight loader scheduling
- Accumulator register allocation
- On-chip memory optimization

## Hardware Target

VIS VPU with the following instruction types:
- `OffchipDataLoader`: Load data from off-chip memory
- `DataLoader`: Load data into line buffers
- `WeightLoader`: Load convolution weights
- `OffsetLoader`: Load deformable convolution offsets
- `QuantLoader`: Load quantization parameters
- `DataStorer`: Store computed results
- `OffchipDataStorer`: Store data to off-chip memory

## License

Apache License 2.0

## References

- [Apache TVM](https://tvm.apache.org/)
- [TVM Documentation](https://tvm.apache.org/docs/)
- [Relay IR](https://tvm.apache.org/docs/arch/relay_intro.html)

