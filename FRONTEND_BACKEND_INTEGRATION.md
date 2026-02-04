# Frontend-Backend Integration Guide

This guide explains how to connect the TVM frontend (Relay/TIR) with the VIS VPU instruction backend.

## Table of Contents

1. [Overview](#overview)
2. [Compilation Pipeline](#compilation-pipeline)
3. [Two Approaches](#two-approaches)
4. [USRNet-Specific Instructions](#usrnet-specific-instructions)
5. [Quick Start](#quick-start)

## Overview

The compilation stack has three main components:

```
┌─────────────────────────────────────┐
│  1. Frontend (TVM)                  │
│     - ONNX/PyTorch import           │
│     - Relay IR optimization         │
│     - TIR lowering                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Middle Layer (NEW)              │
│     - IR traversal                  │
│     - Operator → Instruction mapping│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. Backend                         │
│     - Instruction generation        │
│     - Binary assembly               │
└─────────────────────────────────────┘
```

## Compilation Pipeline

### Step-by-Step Process

1. **Import Model** (Frontend)
   - Input: ONNX/PyTorch model
   - Output: Relay IR (`01_imported.txt`)
   - Tool: `frontend/onnx_importer.py`

2. **Optimize** (Frontend)
   - Input: Relay IR
   - Output: Optimized Relay IR (`02_optimized.txt`)
   - Tool: `passes/pass_manager.py`

3. **Lower to TIR** (Frontend, optional)
   - Input: Optimized Relay IR
   - Output: TIR (`03_tir.txt`)
   - Tool: TVM's relay.build

4. **Generate Instructions** (Middle Layer - **THIS IS NEW**)
   - Input: Relay IR / TIR
   - Output: Instruction list
   - Tool: `codegen/ir_to_inst_bridge.py`

5. **Package Binary** (Backend)
   - Input: Instructions + Parameters
   - Output: Binary files (`*.bin`, `*.txt`)
   - Tool: `codegen/binary_packer.py` + `assembler.py`

## Two Approaches

### Approach 1: Automated IR Traversal

**Best for:** Standard models, rapid prototyping

**How it works:**
- Automatically traverse Relay IR
- Map each operator to instruction templates
- Generic but may need tuning for complex ops

**Example:**
```python
from examples.compile_usr_net_with_backend import compile_usrnet_automated

outputs = compile_usrnet_automated(
    model_path="USR_Net.onnx",
    output_dir="./output/usr_net"
)
```

**Pros:**
- ✓ Automatic
- ✓ Works for standard operators
- ✓ Easy to use

**Cons:**
- ✗ May not optimize for specific hardware patterns
- ✗ Limited control over instruction sequence
- ✗ Doesn't handle custom ops well

### Approach 2: Manual Backend (Recommended for USRNet)

**Best for:** Complex models, optimal performance, custom operations

**How it works:**
- Analyze model structure from IR
- Manually generate instructions using `sd_sr_codegen.py` patterns
- Full control over instruction sequences

**Example:**
```python
from examples.compile_usr_net_with_backend import compile_usrnet_manual

outputs = compile_usrnet_manual(
    model_path="USR_Net.onnx",
    output_dir="./output/usr_net"
)
```

**Pros:**
- ✓ Full control over instruction generation
- ✓ Optimal for hardware
- ✓ Handles deformable convolutions and custom ops
- ✓ Precise buffer management

**Cons:**
- ✗ Requires manual implementation
- ✗ Need to understand model structure
- ✗ More code to write

## USRNet-Specific Instructions

USRNet has special requirements:

1. **Deformable Convolutions**
   - Requires offset generation layers
   - Uses `OffsetLoader` instructions
   - Special bilinear interpolation

2. **Buffer Management**
   - Ping-pong buffers (a/b)
   - Line buffers (0/1)
   - Accumulator registers

3. **Quantization**
   - Per-layer quantization parameters
   - Multiple quantization modes (0-7)

### Example: Generate USRNet Instructions

```python
from instruction import *

# Clear previous instructions
Inst.code_list = []
Inst.current_code_num = 0

# 1. Load off-chip weights and quantization params
OffchipDataLoader.dispatch(
    transnum=100,
    load_model=0,
    src_buffer_idx=2,  # Quantization buffer
    bas_addr=0
)

# 2. For each convolution layer
layer_idx = 0

# Load quantization parameters
QuantLoader.dispatch(
    quant_reg_load_idx=0,
    quant_mode=0,
    layer_idx=layer_idx,
    transnum=4,
    bas_addr=0
)

# Load input data
DataLoader.dispatch(
    layer_idx=layer_idx,
    line_buffer_reshape=0,
    is_padding_row=0,
    read_mode=0,
    transnum=32,
    line_buffer_idx=0,
    src_buffer_idx='a',
    bas_addr=0
)

# Load weights
WeightLoader.dispatch(
    acc_reg_comp_idx=0,
    kernal_size=0,
    line_buffer_row_shift=1,
    line_buffer_idx=0,
    is_padding_col=1,
    weight_parall_mode=0,
    is_new=0,
    transnum=9,
    bas_addr=0,
    is_bilinear_bicubic=0,
    offset_reg_idx=0
)

# Store results
DataStorer.dispatch(
    quant_config_idx=0,
    pixelshuffle_out_mode=0,
    is_pixelshuffle=0,
    pooling_out_mode=0,
    pooling_out_new=0,
    is_pooling=0,
    reg_out_idx=0,
    acc_mode=0,
    transfer_num=1,
    store_mode=0,
    stride=32,
    base_addr_pooling=0,
    base_addrs_res=0,
    is_bicubic_add=0,
    is_first_or_last_row=0,
    is_mask=0,
    is_new=0,
    dest_buffer_idx='b'
)

# 3. Get generated instructions
instructions = Inst.code_list
```

## Quick Start

### For USRNet Compilation

```bash
cd tvm-design

# Run the manual compilation (recommended)
python examples/compile_usr_net_with_backend.py
```

This will:
1. Import USRNet from ONNX
2. Analyze the model structure
3. Generate hardware instructions
4. Package into binary files

### Output Files

```
output/usr_net/
├── usr_net_inst.txt          # Human-readable instructions
├── usr_net_inst.bin          # Binary instruction format
├── usr_net_params.json       # Parameter metadata
├── 01_imported.txt           # Initial Relay IR
├── 02_optimized.txt          # Optimized Relay IR
└── 03_tir.txt                # TIR representation
```

### Verify Output

```bash
# Check instruction count
wc -l output/usr_net/usr_net_inst.txt

# Check binary size
ls -lh output/usr_net/usr_net_inst.bin

# Verify parameters
cat output/usr_net/usr_net_params.json | head -20
```

## Customization

### Add Custom Operator Support

1. **Frontend**: Register custom op in `frontend/custom_ops/`
2. **Middle**: Add mapping in `codegen/ir_to_inst_bridge.py`
3. **Backend**: Implement instruction sequence in `instruction.py`

### Example: Add Deformable Conv Support

```python
# In codegen/ir_to_inst_bridge.py

def _generate_deformable_conv_instructions(self, layer: Dict):
    """Generate instructions for deformable convolution."""
    from instruction import OffsetLoader, DataLoader, WeightLoader
    
    # 1. Generate offset field
    # (offset generation layer)
    
    # 2. Load offset into register
    OffsetLoader.dispatch(
        offset_reg_idx=0,
        bas_addr=offset_base_addr
    )
    
    # 3. Perform deformable convolution
    WeightLoader.dispatch(
        # ... params ...
        is_bilinear_bicubic=1,  # Enable offset-based sampling
        offset_reg_idx=0
    )
```

## Troubleshooting

### Issue: Empty instruction files

**Symptom:** `usr_net_inst.txt` and `usr_net_inst.bin` are empty

**Cause:** Instruction generation step not executed

**Solution:**
1. Check that `_generate_instructions()` is called in driver
2. Verify IR traversal is finding operators
3. Use manual approach for complex models

### Issue: Unsupported operator

**Symptom:** Warning "No handler for operator: xxx"

**Solution:**
1. Add operator handler in `LayerMapper`
2. Implement lowering in `InstructionLowering`
3. Or use manual approach and implement directly

### Issue: Wrong instruction sequence

**Symptom:** Model runs but gives wrong results

**Solution:**
1. Check buffer management (a/b switching)
2. Verify quantization parameters
3. Check dependency computation
4. Compare with `sd_sr_codegen.py` reference implementation

## Next Steps

1. **Test on simple models first**: Try FSRCNN before USRNet
2. **Verify each stage**: Check outputs at each compilation stage
3. **Use visualization**: Enable pass visualization to understand transformations
4. **Profile performance**: Measure instruction count and cycles
5. **Iterate and optimize**: Refine instruction sequences based on hardware feedback

## References

- `sd_sr_codegen.py`: Reference implementation for SR models
- `instruction.py`: Instruction API documentation
- `examples/compile_usr_net_with_backend.py`: Complete examples
- `codegen/ir_to_inst_bridge.py`: IR traversal implementation

