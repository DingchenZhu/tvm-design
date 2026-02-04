# TVM Architecture & PyTorch Frontend Integration Guide

## Table of Contents
1. [TVM Overview](#tvm-overview)
2. [TVM Architecture Layers](#tvm-architecture-layers)
3. [PyTorch to TVM Pipeline](#pytorch-to-tvm-pipeline)
4. [Frontend Conversion Process](#frontend-conversion-process)
5. [Relay IR Structure](#relay-ir-structure)
6. [Your Compiler Stack](#your-compiler-stack)

---

## 1. TVM Overview

TVM (Tensor Virtual Machine) is a compiler stack for deep learning systems. It provides:
- **Frontend**: Import models from various frameworks (PyTorch, ONNX, TensorFlow)
- **IR (Intermediate Representation)**: Relay IR for high-level optimization
- **Backend**: Code generation for various hardware targets

```
┌─────────────────────────────────────────────────────────────────┐
│                         TVM STACK                               │
├─────────────────────────────────────────────────────────────────┤
│  Frameworks: PyTorch, TensorFlow, ONNX, Keras, MXNet           │
│                            ↓                                     │
│  Frontend: Convert to Relay IR                                  │
│                            ↓                                     │
│  Relay IR: High-level graph optimizations                       │
│                            ↓                                     │
│  TIR (Tensor IR): Low-level tensor operations                   │
│                            ↓                                     │
│  Backend: Code generation (LLVM, CUDA, etc.)                    │
│                            ↓                                     │
│  Hardware: CPU, GPU, TPU, VPU, Custom accelerators             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. TVM Architecture Layers

### 2.1 Directory Structure

```
tvm/
├── python/
│   └── tvm/
│       ├── relay/                 # High-level IR and optimizations
│       │   ├── frontend/          # Model importers
│       │   │   ├── pytorch.py     # PyTorch → Relay converter
│       │   │   ├── onnx.py        # ONNX → Relay converter
│       │   │   ├── tensorflow.py  # TensorFlow → Relay converter
│       │   │   └── ...
│       │   ├── transform/         # Relay optimization passes
│       │   ├── op/                # Relay operators
│       │   └── backend/           # Backend compilation
│       ├── tir/                   # Low-level Tensor IR
│       ├── target/                # Hardware targets
│       ├── auto_scheduler/        # Auto-tuning
│       └── runtime/               # Runtime system
├── src/                           # C++ implementation
│   ├── relay/
│   ├── tir/
│   ├── target/
│   └── runtime/
└── include/                       # C++ headers
```

### 2.2 Compilation Flow

```
┌──────────────┐
│ PyTorch Model│
└──────┬───────┘
       │ torch.jit.trace()
       ↓
┌──────────────┐
│ TorchScript  │ (Graph representation)
└──────┬───────┘
       │ relay.frontend.from_pytorch()
       ↓
┌──────────────┐
│  Relay IR    │ (High-level graph)
└──────┬───────┘
       │ Optimization passes
       ↓
┌──────────────┐
│ Optimized    │
│  Relay IR    │
└──────┬───────┘
       │ relay.build()
       ↓
┌──────────────┐
│  TIR         │ (Low-level tensor ops)
└──────┬───────┘
       │ Code generation
       ↓
┌──────────────┐
│ Target Code  │ (LLVM IR, CUDA, etc.)
└──────┬───────┘
       │ Compilation
       ↓
┌──────────────┐
│ Runtime      │
│ Module       │ (Executable)
└──────────────┘
```

---

## 3. PyTorch to TVM Pipeline

### 3.1 Step-by-Step Process

```python
# Step 1: PyTorch Model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

model = MyModel()
model.eval()
```

```python
# Step 2: Trace to TorchScript
example_input = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(model, example_input)

# TorchScript creates a graph:
# graph(%x : Tensor):
#   %1 : Tensor = aten::conv2d(%x, %weight, %bias, ...)
#   %2 : Tensor = aten::relu(%1)
#   return %2
```

```python
# Step 3: Convert to Relay IR
import tvm
from tvm import relay

input_shape = (1, 3, 224, 224)
input_name = 'input'

mod, params = relay.frontend.from_pytorch(
    scripted_model,
    [(input_name, input_shape)]
)

# Relay IR:
# def @main(%input: Tensor[(1, 3, 224, 224), float32]) {
#   %0 = nn.conv2d(%input, %weight, ...);
#   %1 = nn.relu(%0);
#   %1
# }
```

### 3.2 Detailed Conversion Process

```
PyTorch Model
     │
     │ torch.jit.trace()
     ↓
TorchScript Graph Representation
     │
     │ Graph nodes iteration
     ↓
┌────────────────────────────────────┐
│  relay.frontend.pytorch.py         │
│                                    │
│  GraphImporter class:              │
│  • Iterates through TorchScript    │
│    graph nodes                     │
│  • Maps PyTorch ops to Relay ops   │
│  • Converts constants/parameters   │
│  • Builds Relay expression tree    │
│                                    │
│  Key components:                   │
│  • convert_map: Dict mapping       │
│    aten::op → handler function     │
│  • Handler functions: Convert      │
│    specific operations             │
│  • Type inference: Infer tensor    │
│    shapes and types                │
└────────────────────────────────────┘
     │
     ↓
Relay IR Module
```

---

## 4. Frontend Conversion Process

### 4.1 PyTorch Frontend Structure

```
tvm/python/tvm/relay/frontend/pytorch.py
├── GraphImporter class
│   ├── __init__()           # Initialize converter
│   ├── from_pytorch()       # Entry point
│   ├── convert_operators()  # Convert graph nodes
│   └── convert_map          # Dict: op_name → handler
│
├── Operator Handlers
│   ├── conv2d()             # Handle aten::conv2d
│   ├── relu()               # Handle aten::relu
│   ├── pad_generic()        # Handle aten::pad (we added this!)
│   └── ... (100+ handlers)
│
└── Helper Functions
    ├── _get_constant()      # Extract constants from graph
    ├── _infer_shape()       # Infer tensor shapes
    └── _infer_type()        # Infer tensor types
```

### 4.2 How PyTorch Operators Map to Relay

**Example: aten::conv2d → relay.nn.conv2d**

```python
# In pytorch.py (simplified)
class GraphImporter:
    def __init__(self):
        self.convert_map = {
            "aten::conv2d": self.conv2d,
            "aten::relu": self.relu,
            "aten::pad": self.pad_generic,  # Our addition!
            # ... 100+ more
        }

    def conv2d(self, inputs, input_types):
        """Convert aten::conv2d to relay.nn.conv2d"""
        data = inputs[0]           # Input tensor
        weight = inputs[1]         # Conv weights
        bias = inputs[2]           # Bias (optional)
        stride = inputs[3]         # Stride
        padding = inputs[4]        # Padding
        # ... more parameters

        # Create Relay op
        out = _op.nn.conv2d(
            data, weight,
            strides=stride,
            padding=padding,
            # ... more args
        )

        # Add bias if present
        if bias is not None:
            out = _op.nn.bias_add(out, bias)

        return out
```

### 4.3 Operator Conversion Flow

```
TorchScript Node: aten::conv2d(%input, %weight, %bias, [2,2], [1,1], ...)
                           ↓
        GraphImporter.convert_operators()
                           ↓
        Look up "aten::conv2d" in convert_map
                           ↓
        Call self.conv2d(inputs, input_types)
                           ↓
        Extract parameters from inputs
                           ↓
        Create relay.nn.conv2d expression
                           ↓
Relay Expression: %0 = nn.conv2d(%input, %weight, strides=[2,2], padding=[1,1], ...)
                           ↓
        Add to Relay IR graph
```

### 4.4 Our aten::pad Implementation

```python
# What we added to pytorch.py

# 1. Handler function
def pad_generic(self, inputs, input_types):
    """Convert aten::pad to relay.nn.pad"""
    data = inputs[0]        # Input tensor
    pad_list = inputs[1]    # Padding values [left, right, top, bottom]
    mode = inputs[2]        # 'constant', 'reflect', etc.
    value = inputs[3]       # Fill value for constant padding

    # Convert padding format
    # PyTorch: [left, right, top, bottom]
    # Relay: [[top, bottom], [left, right]]
    relay_padding = convert_padding_format(pad_list)

    # Map mode names
    mode_map = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',
        'circular': 'wrap'
    }

    # Create Relay pad operation
    return _op.nn.pad(
        data, relay_padding,
        pad_value=value,
        pad_mode=mode_map[mode]
    )

# 2. Register in convert_map
self.convert_map = {
    # ... existing operators
    "aten::pad": self.pad_generic,  # NEW!
}
```

---

## 5. Relay IR Structure

### 5.1 What is Relay IR?

Relay is TVM's high-level intermediate representation. It's:
- **Functional**: Based on functional programming concepts
- **Typed**: Strongly typed with shape information
- **Differentiable**: Supports gradient computation
- **Optimizable**: Designed for graph-level optimizations

### 5.2 Relay IR Components

```python
# Relay IR consists of:

# 1. Module (IRModule)
mod = tvm.IRModule()

# 2. Function (defines computation)
func = relay.Function(
    params=[input_var],     # Input parameters
    body=expr,              # Computation expression
    ret_type=tensor_type    # Return type
)

# 3. Expression (computational graph)
# Types of expressions:
# - Var: Variables (%x, %y)
# - Constant: Fixed values
# - Call: Function/operator calls
# - Let: Variable binding
# - If: Conditional
# - Tuple: Multiple values
# - TupleGetItem: Extract from tuple

# 4. Operators
# - nn.conv2d, nn.relu, nn.pad, etc.
# - Defined in relay.op
```

### 5.3 Example Relay IR

**PyTorch Model:**
```python
class Model(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        x = F.conv2d(x, weight, bias)
        x = F.relu(x)
        return x
```

**Relay IR:**
```
def @main(%x: Tensor[(1, 3, 224, 224), float32],
          %weight: Tensor[(64, 3, 3, 3), float32],
          %bias: Tensor[(64), float32]) -> Tensor[(1, 64, 224, 224), float32] {

  // Padding
  %0 = nn.pad(%x, [[0, 0], [0, 0], [1, 1], [1, 1]],
              pad_value=0f, pad_mode="constant")
       /* ty=Tensor[(1, 3, 226, 226), float32] */

  // Convolution
  %1 = nn.conv2d(%0, %weight,
                 padding=[0, 0, 0, 0],
                 channels=64,
                 kernel_size=[3, 3])
       /* ty=Tensor[(1, 64, 224, 224), float32] */

  // Bias add
  %2 = nn.bias_add(%1, %bias)
       /* ty=Tensor[(1, 64, 224, 224), float32] */

  // ReLU
  %3 = nn.relu(%2)
       /* ty=Tensor[(1, 64, 224, 224), float32] */

  %3
}
```

---

## 6. Your Compiler Stack

### 6.1 Your Project Structure

```
tvm-design/
├── frontend/                    # Model importers
│   ├── __init__.py
│   ├── model_loader.py          # Unified loader interface
│   ├── pytorch_importer.py      # PyTorch → Relay
│   └── onnx_importer.py         # ONNX → Relay
│
├── passes/                      # Optimization passes
│   ├── pass_manager.py          # Pass pipeline manager
│   ├── basic_passes.py          # Basic optimizations
│   ├── fusion_pass.py           # Operator fusion
│   └── redundancy_elimination.py
│
├── codegen/                     # Code generation
│   ├── layer_mapper.py          # Map Relay ops to instructions
│   ├── ir_to_inst_bridge.py    # Relay IR → Instructions
│   └── binary_packer.py         # Pack to binary
│
├── compiler/                    # Compiler driver
│   ├── driver.py                # Main compiler class
│   └── config.py                # Configuration
│
├── instruction.py               # Instruction definitions
├── assembler.py                 # Binary assembler
│
└── examples/
    ├── compile_usr_net.py       # ONNX compilation
    └── compile_fsrcnn.py        # PyTorch compilation
```

### 6.2 Your Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR COMPILATION FLOW                        │
└─────────────────────────────────────────────────────────────────┘

PyTorch/ONNX Model
      ↓
┌──────────────────────┐
│  frontend/           │
│  - model_loader.py   │  ← Unified interface
│  - pytorch_importer  │  ← Calls relay.frontend.from_pytorch()
│  - onnx_importer     │  ← Calls relay.frontend.from_onnx()
└──────┬───────────────┘
       │ Imports model
       ↓
  TVM Relay IR Module
       ↓
┌──────────────────────┐
│  passes/             │
│  - pass_manager.py   │  ← Orchestrates optimization
│  - basic_passes.py   │  ← FoldConstant, Simplify, etc.
│  - fusion_pass.py    │  ← Fuse conv+bias+relu
│  - dce_pass.py       │  ← Dead code elimination
└──────┬───────────────┘
       │ Optimizes IR
       ↓
  Optimized Relay IR
       ↓
┌──────────────────────┐
│  codegen/            │
│  - layer_mapper.py   │  ← Map Relay ops to your instructions
│  - ir_to_inst       │  ← Convert IR to instruction list
└──────┬───────────────┘
       │ Generates instructions
       ↓
  Instruction List
  [
    {op: 'DataLoader', ...},
    {op: 'WeightLoader', ...},
    {op: 'Conv', ...},
    ...
  ]
       ↓
┌──────────────────────┐
│  assembler.py        │  ← Encode to binary
│  - Bit packing      │
│  - 32-bit alignment  │
└──────┬───────────────┘
       │ Assembles binary
       ↓
  Binary File (.bin)
       ↓
  VIS VPU Hardware
```

### 6.3 How Your Frontend Connects

**Your Code** (`frontend/pytorch_importer.py`):
```python
class PyTorchImporter:
    def import_model(self, model, input_shapes):
        """Import PyTorch model using TVM's frontend."""

        # Step 1: Trace to TorchScript
        scripted_model = torch.jit.trace(model, example_inputs)

        # Step 2: Use TVM's PyTorch frontend
        mod, params = relay.frontend.from_pytorch(
            scripted_model,
            input_shapes
        )
        # ↑ This calls the code in:
        #   /path/to/tvm/python/tvm/relay/frontend/pytorch.py
        #   which we patched to add aten::pad support!

        return mod, params
```

**Connection Flow:**
```
Your Code                          TVM Code
─────────                          ────────

pytorch_importer.py
    ↓
relay.frontend.from_pytorch()  →  tvm/relay/frontend/pytorch.py
    ↓                                  ↓
                                  GraphImporter class
                                       ↓
                                  convert_operators()
                                       ↓
                                  convert_map lookup
                                       ↓
                                  Handler functions:
                                  - conv2d()
                                  - relu()
                                  - pad_generic() ← We added this!
                                       ↓
                                  relay.nn.* operations
                                       ↓
                                  Returns Relay IR Module
    ↓                                  ↓
Relay IR Module            ←──────────┘
    ↓
passes/pass_manager.py
    ↓
Your optimization passes
    ↓
codegen/layer_mapper.py
    ↓
VIS VPU instructions
```

---

## 7. Key Concepts

### 7.1 TorchScript

**What**: PyTorch's intermediate representation
**Format**: Graph-based (nodes and edges)
**Created by**: `torch.jit.trace()` or `torch.jit.script()`

```python
# Original PyTorch
def forward(x):
    return F.relu(F.conv2d(x, weight))

# TorchScript Graph
graph(%x, %weight):
    %0 = aten::conv2d(%x, %weight, ...)
    %1 = aten::relu(%0)
    return (%1)
```

### 7.2 Relay IR

**What**: TVM's high-level IR
**Format**: Functional (like ML functions)
**Purpose**: Hardware-independent optimization

```python
def @main(%x, %weight):
    %0 = nn.conv2d(%x, %weight, ...)
    %1 = nn.relu(%0)
    %1
```

### 7.3 TIR (Tensor IR)

**What**: TVM's low-level IR
**Format**: Loop-based tensor operations
**Purpose**: Hardware-specific optimization

```python
for i in range(0, 224):
    for j in range(0, 224):
        for k in range(0, 64):
            output[i, j, k] = relu(
                conv_compute(input, weight, i, j, k)
            )
```

### 7.4 Operator Registration

**How operators are registered:**

```python
# In pytorch.py
self.convert_map = {
    "aten::conv2d": self.conv2d,      # PyTorch op → handler
    "aten::relu": self.relu,
    "aten::pad": self.pad_generic,     # Our addition!
}

# When TVM sees "aten::pad" in TorchScript:
# 1. Lookup "aten::pad" in convert_map
# 2. Call self.pad_generic(inputs, input_types)
# 3. Handler returns Relay expression
# 4. Add to Relay IR graph
```

---

## 8. Data Flow Diagrams

### 8.1 Type Information Flow

```
PyTorch Model (dynamic types)
      ↓ torch.jit.trace()
TorchScript (inferred types)
      ↓ from_pytorch()
Relay IR (explicit types)
      - Tensor[(1, 3, 224, 224), float32]
      - All shapes known at compile time
```

### 8.2 Parameter Flow

```
PyTorch Parameters
      ↓
TorchScript Constants
      ↓
_get_constant() in pytorch.py
      ↓
Relay Constants
      ↓
params dict = {
    'weight': numpy array,
    'bias': numpy array,
}
      ↓
Your binary_packer.py
      ↓
Binary parameter file
```

---

## 9. Summary

### Key Points

1. **TVM Structure**:
   - Frontend (import) → Relay IR (optimize) → TIR (lower) → Backend (codegen)

2. **PyTorch Connection**:
   - PyTorch → TorchScript → Relay IR (via pytorch.py frontend)

3. **Operator Conversion**:
   - Each PyTorch op (aten::*) → Handler function → Relay op (nn.*)

4. **Your Integration**:
   - Your code uses TVM's frontend to get Relay IR
   - Then applies custom passes and codegen for VIS VPU

5. **What We Did**:
   - Added `aten::pad` handler to pytorch.py
   - Mapped it to `relay.nn.pad`
   - Registered in convert_map

### Architecture Hierarchy

```
Level 5: Framework APIs (PyTorch, TensorFlow, ONNX)
            ↓
Level 4: TVM Frontend (pytorch.py, onnx.py) ← We modified this!
            ↓
Level 3: Relay IR (High-level graph)
            ↓
Level 2: TIR (Low-level tensors)
            ↓
Level 1: Target Code (LLVM, CUDA, Your VIS VPU)
            ↓
Level 0: Hardware Execution
```

---

## 10. References

- **TVM Docs**: https://tvm.apache.org/docs/
- **Relay Docs**: https://tvm.apache.org/docs/arch/relay_intro.html
- **PyTorch Frontend**: `tvm/python/tvm/relay/frontend/pytorch.py`
- **Your Implementation**: `tvm-design/frontend/pytorch_importer.py`

---

**Created**: February 3, 2026
**Purpose**: Understanding TVM architecture and PyTorch integration
