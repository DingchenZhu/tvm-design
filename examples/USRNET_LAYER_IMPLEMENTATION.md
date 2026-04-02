# USRNet Layer Type Implementation Guide

## Overview

This document describes the complete implementation of layer type handlers for USRNet, following the pattern established in `sd_codegen.py`.

## Architecture

```
Frontend (TVM/Relay)
    ↓
Layer Analysis & IR Traversal
    ↓
Layer Type Dispatch (generate_usrnet_instructions)
    ↓
Individual Layer Handlers (_generate_*_instructions)
    ↓
Backend Instructions (DataLoader, WeightLoader, etc.)
    ↓
Binary Output
```

## Implemented Layer Types

### 1. Standard Convolution (`nn.conv2d`)
**Function**: `_generate_conv_instructions()`

**Instruction Sequence**:
```python
QuantLoader    # Load quantization parameters (scale, zero_point)
DataLoader     # Load input feature maps from buffer
WeightLoader   # Load convolution kernel weights
DataStorer     # Store output and apply quantization
```

**Based on**: `sd_codegen.py` lines 137-210 (layer 0 - basic conv pattern)

**Key Parameters**:
- `layer_idx`: 5-bit hardware layer index (wrapped with modulo 32)
- `quant_mode`: 0 for standard convolution
- `transnum`: Transfer count based on feature map size
- `src_buffer_idx`: Ping-pong between 'a' and 'b'

### 2. Deformable Convolution (`deformable_conv`)
**Function**: `_generate_deformable_conv_instructions()`

**Instruction Sequence**:
```python
QuantLoader     # Load quantization parameters
OffsetLoader    # Load offset parameters for deformable sampling
DataLoader      # Load input features
WeightLoader    # Load weights with is_bilinear_bicubic=1
DataStorer      # Store results
```

**Based on**: Deformable convolution pattern from deep learning literature

**Key Features**:
- Offset prediction for adaptive receptive fields
- Bilinear interpolation for sub-pixel sampling
- Uses `is_bilinear_bicubic=1` in WeightLoader
- `offset_reg_idx` specifies which offset register to use

### 3. Activation Functions
**Function**: `_generate_activation_instructions()`

**Supported**: `nn.relu`, `nn.leaky_relu`, `sigmoid`, `tanh`

**Implementation**:
- Typically fused with preceding convolution
- Controlled by `acc_mode` in DataStorer
- Can be standalone with DataLoader → DataStorer

**Based on**: `sd_codegen.py` implicit fusion in DataStorer

### 4. Element-wise Operations
**Function**: `_generate_elementwise_instructions()`

**Supported**: `add`, `subtract`, `multiply`

**Instruction Sequence**:
```python
DataLoader(line_buffer_idx=0)  # Load first operand
DataLoader(line_buffer_idx=1)  # Load second operand
DataStorer(acc_mode=X)          # Store with operation
```

**acc_mode Values**:
- `0`: Overwrite (no operation)
- `1`: Add (for element-wise addition)
- `2`: Concatenate (for channel concat)
- `3`: Multiply (for element-wise multiplication)

**Based on**: `sd_codegen.py` lines with `acc_mode` variations

### 5. Batch Normalization
**Function**: `_generate_norm_instructions()`

**Supported**: `nn.batch_norm`, `nn.layer_norm`

**Implementation**:
- Fused with convolution during quantization
- BN parameters folded into scale/zero_point
- Loaded via QuantLoader only

**Based on**: `sd_codegen.py` quantization parameter loading

**Mathematical Transformation**:
```
BatchNorm: y = γ * (x - μ) / √(σ² + ε) + β
After folding: scale = γ / √(σ² + ε), bias = β - γμ / √(σ² + ε)
```

### 6. Pooling Layers
**Function**: `_generate_pooling_instructions()`

**Supported**: `nn.max_pool2d`, `nn.avg_pool2d`

**Instruction Sequence**:
```python
DataLoader                  # Load input data
DataStorer(is_pooling=1)    # Store with pooling enabled
```

**Key Parameters**:
- `is_pooling=1`: Enable pooling operation
- `pooling_out_mode`:
  - `0`: Max pooling
  - `1`: Average pooling
  - `2`: Other pooling types
- `stride`: Controls pooling window stride

**Based on**: `sd_codegen.py` lines 467-526 (layer 2 with pooling)

### 7. Concatenation
**Function**: `_generate_concat_instructions()`

**Supported**: `concatenate`

**Implementation**:
- No explicit instruction generated
- Handled via buffer address management
- Uses buffer models (buffer_a_model, buffer_b_model)

**Based on**: `sd_codegen.py` lines 606-607, 791-792 buffer management pattern

**Example from sd_codegen.py**:
```python
datastorermanager.buffer_a_model.append({
    "begin": 0, 
    "end": 144*4*2, 
    "info": "c1_for_cat", 
    "valid": True
})
```

### 8. Upsampling/Resizing
**Function**: `_generate_upsample_instructions()`

**Supported**: `nn.upsampling`, `image.resize2d`

**Methods**:
1. **Pixel Shuffle** (depth_to_space):
   - `is_pixelshuffle=1`
   - `pixelshuffle_out_mode` controls rearrangement
   
2. **Bilinear/Bicubic Interpolation**:
   - `is_bicubic_add=1`
   - Hardware-accelerated interpolation
   
3. **Nearest Neighbor** (default):
   - Standard upsampling without interpolation

**Based on**: `sd_codegen.py` lines 1433 (pixel shuffle), 1144 (bilinear)

### 9. Clip Operation
**Function**: `_generate_clip_instructions()`

**Supported**: `clip`

**Implementation**:
- Value clamping between min and max
- Handled by quantization parameters
- No special instruction needed

**Based on**: Quantization clipping in sd_codegen.py

### 10. Layout Transformations
**Function**: `_generate_layout_transform_instructions()`

**Supported**: `reshape`, `transpose`, `squeeze`, `expand_dims`

**Key Parameters**:
- `line_buffer_reshape`: Controls input reshape
  - `0`: No reshape
  - `1`: Transpose
  - `2`: Channel-wise reshape
  - `3`: Custom reshape
- `store_mode`: Controls output layout
  - `0`: Standard H-major order
  - `1`: Sequential channel-major
  - `2`: W-major order
  - `3`: Other layouts

**Based on**: `sd_codegen.py` lines 644, 736, 1211 (line_buffer_reshape usage)

## Buffer Management Strategy

Following `sd_codegen.py` pattern:

### Ping-Pong Buffering
```python
Layer N:   Read from buffer 'a' → Write to buffer 'b'
Layer N+1: Read from buffer 'b' → Write to buffer 'a'
```

### Buffer Models
Tracks memory regions for feature maps:
```python
{
    "begin": start_address,
    "end": end_address,
    "info": "layer_name_output",
    "valid": True/False  # Indicates if data is still needed
}
```

### Concatenation via Buffer Layout
Instead of explicit concat instruction:
1. Store different feature maps at adjacent addresses
2. Later layers read from concatenated region
3. Buffer model tracks the concatenated layout

## Quantization Modes

Following `sd_codegen.py` patterns (lines 613-622, 704-712):

| Mode | Bit Width | Channels | Use Case |
|------|-----------|----------|----------|
| 0    | 8-bit     | 1-4      | Early conv layers |
| 1    | 8-bit     | 8        | Mid layers |
| 2    | 8-bit     | 16       | Deeper layers |
| 3    | 8-bit     | 32       | Very deep layers |
| 4    | 8-bit     | 32       | Group conv layers |
| 5    | 16-bit    | Variable | High precision |
| 6    | 8-bit     | Variable | Output layers |
| 7    | 8-bit     | Variable | Upsampling layers |

## Layer Index Wrapping

Hardware constraint: Layer index is 5-bit (0-31)

**Solution**:
```python
hw_layer_idx = layer_idx % 32
```

For models with >32 layers, indices wrap around. This is acceptable because:
1. Instructions execute sequentially
2. Layer context maintained by surrounding instructions
3. Dependencies prevent out-of-order execution

## Example Usage

### Complete Layer Processing Pattern

```python
# 1. Analyze model structure
layer_info = analyze_usrnet_structure(mod, params)

# 2. Generate instructions for each layer
for idx, layer in enumerate(layer_info):
    if 'conv2d' in layer['type']:
        _generate_conv_instructions(idx, layer)
    elif 'max_pool' in layer['type']:
        _generate_pooling_instructions(idx, layer)
    # ... other layer types

# 3. Get generated instructions
instructions = Inst.code_list

# 4. Add dependencies (automatic via dispatch)
# Dependencies computed based on:
# - Buffer usage (line_buffer_idx, src_buffer_idx)
# - Register usage (acc_reg_idx, quant_config_idx)
# - Instruction ordering
```

### Instruction Dependency Pattern

From `sd_codegen.py` lines 2417-2518:

```python
WeightLoader depends on:
  1. Previous DataLoader (same line_buffer_idx)
  2. Previous DataStorer (same acc_reg_idx)
  3. Previous WeightLoader (sequential weight loading)

DataStorer depends on:
  1. Previous QuantLoader (same quant_config_idx)
  2. Previous WeightLoader (same acc_reg_idx)
  3. Previous DataStorer (sequential execution)
```

## Hardware Instruction Format

Each instruction type has specific bit field encodings:

### DataLoader Fields (from instruction.py)
- `layer_idx`: 5 bits
- `line_buffer_reshape`: 2 bits
- `is_padding_row`: 3 bits
- `read_mode`: 2 bits
- `transnum`: 4 bits (0-15)
- `line_buffer_idx`: 1 bit
- `src_buffer_idx`: 2 bits
- `bas_addr`: Variable bits

### WeightLoader Fields
- `acc_reg_comp_idx`: 1 bit
- `kernal_size`: 2 bits
- `line_buffer_row_shift`: 3 bits
- `line_buffer_idx`: 1 bit
- `is_padding_col`: 3 bits
- `weight_parall_mode`: 2 bits
- `is_new`: 1 bit
- `transnum`: Variable
- `is_bilinear_bicubic`: 1 bit
- `offset_reg_idx`: Variable

### DataStorer Fields
- `quant_config_idx`: 1 bit
- `pixelshuffle_out_mode`: 2 bits
- `is_pixelshuffle`: 1 bit
- `pooling_out_mode`: 2 bits
- `is_pooling`: 1 bit
- `reg_out_idx`: 1 bit
- `acc_mode`: 3 bits
- `transfer_num`: Variable
- `store_mode`: 2 bits
- `stride`: Variable
- `is_bicubic_add`: 1 bit
- `is_first_or_last_row`: 2 bits
- `is_new`: 1 bit

## Comparison with sd_codegen.py

### Similarities
1. **Instruction sequence pattern**: Same 4-step pattern for convolution
2. **Buffer management**: Ping-pong strategy between buffers
3. **Quantization approach**: Per-layer quantization parameters
4. **Dependency handling**: Automatic computation based on resource usage

### Differences
1. **Model-specific**: sd_codegen.py is U-Net specific; our implementation is generic
2. **Layer coverage**: We support more diverse layer types
3. **Abstraction level**: Our implementation is more modular and reusable
4. **Documentation**: Explicit layer type dispatch vs. inline implementation

## Testing Recommendations

### 1. Per-Layer Testing
Test each layer type handler individually:
```python
layer = {'type': 'nn.conv2d', 'attrs': {...}, 'shape': [1,64,32,32]}
_generate_conv_instructions(0, layer)
assert len(Inst.code_list) == 4  # Quant, Data, Weight, Storer
```

### 2. Sequence Testing
Test multi-layer sequences:
```python
layers = [conv_layer, relu_layer, pool_layer]
for idx, layer in enumerate(layers):
    dispatch_layer(idx, layer)
verify_dependencies(Inst.code_list)
```

### 3. Buffer Management Testing
Verify buffer allocation:
```python
assert buffers_dont_overlap()
assert ping_pong_correct()
assert concat_addresses_sequential()
```

### 4. Hardware Constraints
Verify hardware limits:
```python
for inst in instructions:
    assert inst['layer_idx'] < 32
    assert inst['transnum'] < 16
    # ... other constraints
```

## References

- **sd_codegen.py**: Lines 1-2634 - Complete U-Net implementation
- **instruction.py**: Instruction class definitions and dispatch methods
- **Backend Hardware Spec**: Field widths and constraints

## Summary

This implementation provides complete layer type coverage for USRNet based on the proven patterns from `sd_codegen.py`. The modular design allows for:

1. **Easy extension**: Add new layer types by implementing handler functions
2. **Maintainability**: Clear separation between layer types
3. **Verification**: Each handler can be tested independently
4. **Flexibility**: Adapt to hardware changes by modifying individual handlers

All layer types follow the same basic pattern while allowing for specific optimizations and hardware features unique to each operation type.

