# PyTorch Pad Operator Implementation for TVM

## Summary

We successfully implemented the `aten::pad` operator for TVM's PyTorch frontend. However, there are additional complications with the FSRCNN model related to list constants.

## What Was Done

### 1. ✅ Implemented `aten::pad` Operator

**File**: `/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py`

**Added Function** (after line 1741):
```python
def pad_generic(self, inputs, input_types):
    """
    Generic pad operator that supports multiple modes.

    PyTorch signature: pad(input, pad, mode='constant', value=0)
    """
    data = inputs[0]

    # Extract pad list
    if isinstance(inputs[1], list):
        pad_list = inputs[1]
    else:
        pad_list = list(self.infer_shape(inputs[1]))

    # Extract mode (default to 'constant')
    if len(inputs) > 2 and inputs[2] is not None:
        mode = inputs[2]
        if hasattr(mode, 'data'):
            mode = str(mode.data.numpy().decode('utf-8'))
        elif isinstance(mode, str):
            mode = mode
        else:
            mode = 'constant'
    else:
        mode = 'constant'

    # Extract value (default to 0)
    pad_value = inputs[3] if len(inputs) > 3 and inputs[3] is not None else 0.0

    # Initialize paddings
    pad_len = len(self.infer_shape(data)) * 2
    paddings = [0] * pad_len

    # PyTorch pad format: (left, right, top, bottom, front, back)
    if len(pad_list) >= 2:
        paddings[-1] = pad_list[1]  # right
        paddings[-2] = pad_list[0]  # left
    if len(pad_list) >= 4:
        paddings[-3] = pad_list[3]  # bottom
        paddings[-4] = pad_list[2]  # top
    if len(pad_list) >= 6:
        paddings[-5] = pad_list[5]  # back
        paddings[-6] = pad_list[4]  # front

    # Group into tuple of 2 ints
    paddings = [paddings[i : i + 2] for i in range(0, len(paddings), 2)]

    # Convert to constants
    const_paddings = []
    for pad in paddings:
        const_paddings.append([])
        for p in pad:
            if not isinstance(p, int):
                p = int(_infer_value(p, {}).numpy())
            const_paddings[-1].append(p)

    # Map PyTorch mode names to TVM mode names
    mode_map = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',
        'circular': 'wrap'
    }

    tvm_mode = mode_map.get(mode, 'constant')

    # Apply padding
    if tvm_mode == 'constant':
        return _op.nn.pad(data, const_paddings, pad_value=pad_value, pad_mode=tvm_mode)
    else:
        return _op.nn.pad(data, const_paddings, pad_mode=tvm_mode)
```

**Added Operator Registration** (around line 2950):
```python
"aten::pad": self.pad_generic,
```

### 2. ⚠️ ListType Constant Handling (Partial)

Attempted to add ListType constant handling but encountered complications with how PyTorch TorchScript represents list constants.

## Current Status

✅ **`aten::pad` operator is implemented and registered**
⚠️ **FSRCNN compilation has additional issues**:
- ListType constant extraction is complex
- May require model-specific workarounds

## Backup Files

- Original file: `/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py.backup`
- After aten::pad: `/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py.backup2`

## To Restore Original

```bash
cp /home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py.backup \
   /home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py
```

## Alternative Approaches for FSRCNN

Since FSRCNN uses custom DeformableConv2d and complex operations:

### Option 1: Export to ONNX First

```python
# In your model script
import torch
import torch.onnx

model = FSRCNN(...)
model.eval()
dummy_input = torch.randn(1, 1, 270, 480)

torch.onnx.export(
    model,
    dummy_input,
    "fsrcnn.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)
```

Then compile the ONNX model:
```python
from compiler import TVMVISCompiler, CompilerConfig

config = CompilerConfig(
    output_dir='./output/fsrcnn',
    output_name='fsrcnn',
    dump_ir=True
)

compiler = TVMVISCompiler(config=config)
result = compiler.compile(
    model='fsrcnn.onnx',
    model_format='onnx'
)
```

### Option 2: Use Standard Convolutions

Replace DeformableConv2d with standard Conv2d if possible:
```python
# Instead of:
# DeformableConv2d(s, s, kernel_size=3, padding=1)

# Use:
# nn.Conv2d(s, s, kernel_size=3, padding=1)
```

### Option 3: Implement DeformableConv2d for TVM

If deformable convolution is critical, implement it as a custom TVM operator.

## Testing the aten::pad Implementation

To test just the pad operator with a simple model:

```python
import torch
import torch.nn as nn

class SimplePadModel(nn.Module):
    def forward(self, x):
        # Test different padding modes
        x1 = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        x2 = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        return x1 + x2

model = SimplePadModel()
model.eval()

from compiler import TVMVISCompiler, CompilerConfig

config = CompilerConfig(output_dir='./output/pad_test', output_name='pad_test')
compiler = TVMVISCompiler(config=config)

example_input = torch.randn(1, 3, 32, 32)
result = compiler.compile(
    model=model,
    model_format='pytorch',
    example_inputs=example_input
)
```

## Tools Created

1. **apply_pad_patch.py** - Automatic patch script for aten::pad
2. **pytorch_pad_patch.py** - Documentation and manual patch instructions
3. **apply_listtype_patch.py** - Attempt to handle ListType constants

## Recommendations

1. ✅ **Use the ONNX export approach for FSRCNN**
   - More stable conversion path
   - Better support for custom operations

2. **For future PyTorch models**:
   - Use standard operations when possible
   - Test with simple models first
   - Consider ONNX as intermediate format

3. **If aten::pad issues persist**:
   - Check TVM version compatibility
   - Report issue to TVM community with minimal reproducible example

## Support

If you encounter issues:
1. Check TVM logs for specific operator errors
2. Try simplifying the model
3. Use ONNX export as fallback
4. Restore backups if needed

---

**Created**: February 3, 2026
**Status**: `aten::pad` implemented, FSRCNN needs alternative approach
