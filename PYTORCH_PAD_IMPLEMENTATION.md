# Implementing aten::pad in TVM PyTorch Frontend

## Problem

```
NotImplementedError: The following operators are not implemented: ['aten::pad']
```

## Solution: Two Approaches

### Approach 1: Monkey Patch (Quick, No TVM Rebuild)

This approach patches TVM at runtime without modifying source files.

#### Step 1: Create the patch file

Create `frontend/pytorch_pad_support.py`:

```python
"""Monkey patch to add aten::pad support to TVM PyTorch frontend."""

def patch_pytorch_frontend():
    """Add aten::pad operator support."""
    try:
        from tvm.relay.frontend import pytorch
        from tvm import relay
        from tvm.relay import op as _op
        
        converter_class = pytorch.PyTorchOpConverter
        
        # Define the _pad method
        def _pad(self, inputs, input_types):
            """Implement aten::pad operator.
            
            PyTorch signature:
                torch.nn.functional.pad(input, pad, mode='constant', value=0)
            
            Args:
                inputs[0]: data tensor
                inputs[1]: pad tuple [left, right, top, bottom, ...]
                inputs[2]: mode ('constant', 'reflect', 'replicate', 'circular')
                inputs[3]: value (for constant padding)
            """
            data = inputs[0]
            
            # Extract padding list
            if isinstance(inputs[1], list):
                pad_list = inputs[1]
            else:
                # Try to get from constant
                pad_list = self.infer_shape(inputs[1])
            
            # Get mode (default: 'constant')
            mode = 'constant'
            if len(inputs) > 2 and inputs[2] is not None:
                mode = inputs[2]
            
            # Get fill value (default: 0)
            pad_value = 0.0
            if len(inputs) > 3 and inputs[3] is not None:
                pad_value = inputs[3]
                if isinstance(pad_value, relay.expr.Constant):
                    pad_value = pad_value.data.numpy().item()
            
            # Get input shape
            data_shape = self.infer_shape(data)
            ndim = len(data_shape)
            
            # Initialize pad_width with zeros
            pad_width = [[0, 0]] * ndim
            
            # PyTorch pad format: [left, right, top, bottom, front, back]
            # Applied to last dimensions first
            if len(pad_list) >= 2:
                # Pad last dimension (Width)
                pad_width[-1] = [int(pad_list[0]), int(pad_list[1])]
            
            if len(pad_list) >= 4:
                # Pad second-to-last dimension (Height)
                pad_width[-2] = [int(pad_list[2]), int(pad_list[3])]
            
            if len(pad_list) >= 6:
                # Pad third-to-last dimension (Depth/Channel)
                pad_width[-3] = [int(pad_list[4]), int(pad_list[5])]
            
            # Convert PyTorch mode to Relay mode
            mode_map = {
                'constant': 'constant',
                'reflect': 'reflect',
                'replicate': 'edge',      # Relay uses 'edge' for replicate
                'circular': 'wrap'
            }
            relay_mode = mode_map.get(mode, 'constant')
            
            # Create Relay pad operation
            if relay_mode == 'constant':
                return _op.nn.pad(data, pad_width=pad_width, pad_value=pad_value)
            else:
                return _op.nn.pad(data, pad_width=pad_width, pad_mode=relay_mode)
        
        # Attach the method to the class
        converter_class._pad = _pad
        
        # Patch __init__ to add to convert_map
        original_init = converter_class.__init__
        
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.convert_map["aten::pad"] = self._pad
        
        converter_class.__init__ = patched_init
        
        print("✓ TVM PyTorch frontend patched with aten::pad support")
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch: {e}")
        import traceback
        traceback.print_exc()
        return False
```

#### Step 2: Use the patch in your compilation script

```python
# At the top of your compile script
import sys
sys.path.append('..')

# Import and apply the patch BEFORE importing TVM
from frontend.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

# Now import TVM and compile
import tvm
from tvm import relay
from compiler.driver import TVMVISCompiler

# Your compilation code here
compiler = TVMVISCompiler()
result = compiler.compile(model_path, model_format='pytorch')
```

### Approach 2: Permanent Fix (Modify TVM Source)

This approach permanently adds the operator to TVM.

#### Step 1: Locate the PyTorch frontend file

```bash
cd /home/hansz/scratch-data/design/tvm
# The file is at:
# python/tvm/relay/frontend/pytorch.py
```

#### Step 2: Add the _pad method

Find the `PyTorchOpConverter` class (around line 500-1000) and add:

```python
def _pad(self, inputs, input_types):
    """Operator converter for aten::pad."""
    data = inputs[0]
    
    # Get padding list
    if isinstance(inputs[1], list):
        pad_list = inputs[1]
    else:
        pad_list = self.infer_shape(inputs[1])
    
    # Get mode (default: constant)
    mode = 'constant'
    if len(inputs) > 2 and inputs[2] is not None:
        mode = inputs[2]
    
    # Get pad value (default: 0)
    pad_value = 0.0
    if len(inputs) > 3 and inputs[3] is not None:
        pad_value = inputs[3]
        if isinstance(pad_value, _expr.Constant):
            pad_value = pad_value.data.numpy().item()
    
    # Get shape
    data_shape = self.infer_shape(data)
    ndim = len(data_shape)
    
    # Build pad_width
    pad_width = [[0, 0]] * ndim
    
    if len(pad_list) >= 2:
        pad_width[-1] = [int(pad_list[0]), int(pad_list[1])]
    if len(pad_list) >= 4:
        pad_width[-2] = [int(pad_list[2]), int(pad_list[3])]
    if len(pad_list) >= 6:
        pad_width[-3] = [int(pad_list[4]), int(pad_list[5])]
    
    # Mode mapping
    mode_map = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',
        'circular': 'wrap'
    }
    relay_mode = mode_map.get(mode, 'constant')
    
    # Create pad op
    if relay_mode == 'constant':
        return _op.nn.pad(data, pad_width=pad_width, pad_value=pad_value)
    else:
        return _op.nn.pad(data, pad_width=pad_width, pad_mode=relay_mode)
```

#### Step 3: Register in convert_map

Find the `convert_map` dictionary (around line 3500-3800) and add:

```python
"aten::pad": self._pad,
```

#### Step 4: Rebuild TVM (if using C++ extensions)

```bash
cd /home/hansz/scratch-data/design/tvm/build
cmake ..
make -j$(nproc)

# Or just restart Python if no C++ changes
```

## Understanding PyTorch Pad

### PyTorch Padding Format

```python
torch.nn.functional.pad(input, pad, mode='constant', value=0)
```

- **pad**: Tuple specifying padding for each dimension
- Format: `[left, right, top, bottom, front, back]`
- Applied to **last dimensions first**

### Examples

```python
# 2D tensor (H, W)
x = torch.randn(10, 10)
torch.nn.functional.pad(x, (1, 2, 3, 4))
# Pads: W=[1 left, 2 right], H=[3 top, 4 bottom]
# Output shape: (10+3+4, 10+1+2) = (17, 13)

# 4D tensor (N, C, H, W)
x = torch.randn(1, 3, 10, 10)
torch.nn.functional.pad(x, (1, 1, 2, 2))
# Pads: W=[1, 1], H=[2, 2]
# Output shape: (1, 3, 14, 12)
```

### Relay Padding Format

```python
relay.nn.pad(data, pad_width, pad_value=0, pad_mode='constant')
```

- **pad_width**: `[[before_1, after_1], [before_2, after_2], ...]`
- Specifies padding for **each dimension in order**

### Conversion Example

```python
# PyTorch: pad last 2 dims of 4D tensor (N,C,H,W)
pytorch_pad = [1, 2, 3, 4]  # [left, right, top, bottom]

# Relay: specify padding for all 4 dimensions
relay_pad_width = [
    [0, 0],  # N: no padding
    [0, 0],  # C: no padding
    [3, 4],  # H: top=3, bottom=4
    [1, 2]   # W: left=1, right=2
]
```

## Testing

### Test Script

```python
import torch
import sys
sys.path.append('..')

# Apply patch
from frontend.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

import tvm
from tvm import relay

# Test model
class PadModel(torch.nn.Module):
    def forward(self, x):
        # Pad: left=1, right=1, top=2, bottom=2
        return torch.nn.functional.pad(x, (1, 1, 2, 2), 'constant', 0)

model = PadModel()
model.eval()

# Test input
x = torch.randn(1, 3, 10, 10)

# Trace
scripted = torch.jit.trace(model, x)

# Convert to Relay
try:
    mod, params = relay.frontend.from_pytorch(
        scripted,
        [("input", (1, 3, 10, 10))]
    )
    print("✓ Success! Pad operator converted")
    print("\nGenerated IR:")
    print(mod)
except Exception as e:
    print(f"✗ Failed: {e}")
```

### Expected Output

```
✓ TVM PyTorch frontend patched with aten::pad support
✓ Success! Pad operator converted

Generated IR:
def @main(%input: Tensor[(1, 3, 10, 10), float32]) {
  nn.pad(%input, pad_width=[[0, 0], [0, 0], [2, 2], [1, 1]], pad_value=0f)
}
```

## Complete Example for Your compile_fsrcnn.py

```python
#!/usr/bin/env python3
"""Compile FSRCNN with pad support."""

import sys
sys.path.append('..')

# IMPORTANT: Apply patch BEFORE importing TVM
from frontend.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

import logging
from compiler.driver import TVMVISCompiler
from compiler.config import CompilerConfig, PassConfig

logging.basicConfig(level=logging.INFO)

def main():
    # Configuration
    MODEL_PATH = "../models/fsrcnn.pth"  # or .onnx
    OUTPUT_DIR = "./output/fsrcnn"
    
    # Create compiler
    pass_config = PassConfig(
        opt_level=3,
        enable_fold_constant=True,
        enable_fold_scale_axis=True
    )
    
    config = CompilerConfig(
        target='vis_vpu',
        pass_config=pass_config,
        output_dir=OUTPUT_DIR,
        visualize_passes=False
    )
    
    compiler = TVMVISCompiler(config)
    
    # Compile (pad is now supported!)
    try:
        result = compiler.compile(
            MODEL_PATH,
            model_format='pytorch',  # or 'onnx'
        )
        
        print("\n✓ Compilation successful!")
        print(f"Instructions: {result['num_instructions']}")
        print(f"Parameters: {result['num_parameters']}")
        
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```

## Troubleshooting

### Issue: "AttributeError: 'PyTorchOpConverter' object has no attribute '_pad'"

**Cause**: Patch didn't apply correctly

**Solution**: Ensure patch is called before TVM import

```python
# WRONG
import tvm
patch_pytorch_frontend()  # Too late!

# CORRECT
patch_pytorch_frontend()  # Before TVM
import tvm
```

### Issue: "Still getting NotImplementedError"

**Cause**: Using cached/old TVM import

**Solution**: Restart Python interpreter or clear cache

```bash
# Remove cached imports
rm -rf __pycache__
rm -rf ../__pycache__

# Restart Python
python compile_fsrcnn.py
```

### Issue: "Wrong output shape"

**Cause**: Padding conversion error

**Solution**: Check padding dimensions match:

```python
# Debug: Print padding values
print(f"PyTorch pad: {pad_list}")
print(f"Relay pad_width: {pad_width}")
```

## Summary

**Quick Fix (Recommended for testing)**:
1. Create `frontend/pytorch_pad_support.py` with patch function
2. Call `patch_pytorch_frontend()` at start of compilation script
3. Compile your model

**Permanent Fix (For production)**:
1. Edit `/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py`
2. Add `_pad()` method to `PyTorchOpConverter` class
3. Add `"aten::pad": self._pad` to `convert_map`
4. Rebuild TVM

Choose the monkey patch approach for immediate testing, then submit a proper fix to TVM upstream!

