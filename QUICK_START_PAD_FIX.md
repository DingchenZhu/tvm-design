# Quick Start: Fixing aten::pad Error

## The Problem

```
NotImplementedError: The following operators are not implemented: ['aten::pad']
```

When compiling PyTorch models (like FSRCNN) to TVM, you hit this error because TVM's PyTorch frontend doesn't have the `aten::pad` operator implemented.

## The Solution (3 Steps)

### Step 1: Verify the patch file exists

Check that you have:
```bash
cd /home/hansz/scratch-data/design/tvm-design
ls -la patches/pytorch_pad_support.py
```

If the file exists, you're ready. If not, it should have been created as part of this guide.

### Step 2: Test the patch

Run a quick test to verify it works:

```bash
cd patches
python3 -c "
import sys
sys.path.append('..')
from pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

import torch
import tvm
from tvm import relay

class TestModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.pad(x, (1, 1, 2, 2), 'constant', 0)

model = TestModel()
model.eval()
x = torch.randn(1, 3, 10, 10)

try:
    scripted = torch.jit.trace(model, x)
    mod, params = relay.frontend.from_pytorch(scripted, [('input', (1, 3, 10, 10))])
    print('✓ SUCCESS: aten::pad patch works!')
except Exception as e:
    print(f'✗ FAILED: {e}')
"
```

Expected output:
```
✓ TVM PyTorch frontend patched with aten::pad support
✓ SUCCESS: aten::pad patch works!
```

### Step 3: Use in your compilation

#### Option A: Update existing compile_fsrcnn.py

Add these lines **at the very top** of your `examples/compile_fsrcnn.py`:

```python
import sys
sys.path.append('..')

# ADD THESE TWO LINES BEFORE OTHER IMPORTS
from patches.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

# Now the rest of your imports
import logging
from compiler.driver import TVMVISCompiler
# ... etc
```

#### Option B: Use the provided compile_fsrcnn_with_pad.py

```bash
cd examples
python3 compile_fsrcnn_with_pad.py
```

This script already has the patch applied.

## How It Works

### Before the patch:

```
Your Script
    ↓
Import TVM
    ↓
Load PyTorch model
    ↓
Convert to TVM
    ↓
✗ ERROR: aten::pad not implemented
```

### After the patch:

```
Your Script
    ↓
Apply patch (adds aten::pad support)
    ↓
Import TVM (now knows about aten::pad)
    ↓
Load PyTorch model
    ↓
Convert to TVM
    ↓
✓ SUCCESS: pad operator converted!
```

## What the Patch Does

The patch adds the missing `aten::pad` operator to TVM's PyTorch frontend by:

1. **Defining the conversion function** that translates PyTorch pad to Relay pad
2. **Registering it** in the operator conversion map
3. **Does NOT modify TVM source** - works at runtime!

### PyTorch vs Relay Padding

**PyTorch format:**
```python
torch.nn.functional.pad(x, (left, right, top, bottom))
# Example: (1, 2, 3, 4) means:
#   - Pad width: left=1, right=2
#   - Pad height: top=3, bottom=4
```

**Relay format:**
```python
relay.nn.pad(x, pad_width=[[0,0], [0,0], [3,4], [1,2]])
# Format: [[dim0_before, dim0_after], [dim1_before, dim1_after], ...]
# For 4D (N,C,H,W): N and C have no padding, H has [3,4], W has [1,2]
```

The patch automatically converts between these formats!

## Supported Padding Modes

The patch supports all PyTorch padding modes:

| PyTorch Mode | Relay Mode | Description |
|--------------|------------|-------------|
| 'constant' | 'constant' | Fill with constant value |
| 'reflect' | 'reflect' | Mirror reflection |
| 'replicate' | 'edge' | Replicate edge values |
| 'circular' | 'wrap' | Circular/wrap padding |

## Testing Different Padding Types

### Test Constant Padding

```python
import torch
x = torch.randn(1, 3, 10, 10)
output = torch.nn.functional.pad(x, (1, 1, 2, 2), 'constant', 0)
# Input:  (1, 3, 10, 10)
# Output: (1, 3, 14, 12)  # 10+2+2=14, 10+1+1=12
```

### Test Reflect Padding

```python
output = torch.nn.functional.pad(x, (1, 1, 1, 1), 'reflect')
# Uses mirror reflection at boundaries
```

### Test Replicate Padding

```python
output = torch.nn.functional.pad(x, (2, 2, 2, 2), 'replicate')
# Replicates edge values
```

## Troubleshooting

### Issue: Still getting NotImplementedError

**Cause**: Patch not applied or applied after TVM import

**Solution**: Make sure patch is called **BEFORE** importing TVM

```python
# ✗ WRONG ORDER
import tvm
patch_pytorch_frontend()  # Too late!

# ✓ CORRECT ORDER
patch_pytorch_frontend()  # Before TVM!
import tvm
```

### Issue: "ModuleNotFoundError: No module named 'patches'"

**Cause**: Wrong directory or path issue

**Solution**: Make sure you're running from the right directory or add path:

```python
import sys
sys.path.append('..')  # Go up one directory to find patches/
from patches.pytorch_pad_support import patch_pytorch_frontend
```

### Issue: Patch seems to work but compilation still fails

**Cause**: Different error, not related to pad

**Solution**: Read the actual error message carefully. The pad issue is fixed, but there might be other missing operators.

### Issue: "AttributeError: 'NoneType' object has no attribute 'data'"

**Cause**: Trying to extract padding from unsupported type

**Solution**: This usually means the padding parameter is dynamic. Ensure padding is constant in your model:

```python
# ✓ Good - constant padding
def forward(self, x):
    return F.pad(x, (1, 1, 2, 2))  # Constant values

# ✗ Bad - dynamic padding
def forward(self, x, pad_amount):
    return F.pad(x, (pad_amount, pad_amount))  # Variable
```

## Verifying Success

After compilation, you should see:

```bash
output/fsrcnn/
├── fsrcnn_inst.txt          # Non-empty!
├── fsrcnn_inst.bin          # Non-empty!
├── fsrcnn_params.json       # Parameters
├── 01_imported.txt          # Initial IR (should contain nn.pad)
└── 02_optimized.txt         # Optimized IR
```

Check the IR files:

```bash
# Should show nn.pad operator
grep "pad" output/fsrcnn/01_imported.txt
```

## Complete Working Example

Here's a complete minimal example you can run:

```python
#!/usr/bin/env python3
import sys
sys.path.append('..')

# Apply patch
from patches.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()

# Now use TVM
import torch
import tvm
from tvm import relay

# Create model with padding
class MyModel(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 2, 2))
        return x

model = MyModel()
model.eval()

# Convert to TVM
x = torch.randn(1, 3, 10, 10)
scripted = torch.jit.trace(model, x)

mod, params = relay.frontend.from_pytorch(
    scripted,
    [("input", (1, 3, 10, 10))]
)

print("Success! Pad operator converted:")
print(mod)
```

Save this as `test_pad_simple.py` and run:

```bash
python3 test_pad_simple.py
```

## Next Steps

After fixing the pad issue:

1. **Verify FSRCNN compiles successfully**
   ```bash
   cd examples
   python3 compile_fsrcnn_with_pad.py
   ```

2. **Check for other missing operators**
   - If you hit other NotImplementedError, repeat similar process
   - Common missing ops: `aten::grid_sample`, `aten::pixel_shuffle`

3. **Optimize and test**
   - Verify instruction count is reasonable
   - Test on hardware or simulator

4. **Consider permanent fix**
   - Submit patch to TVM upstream
   - Or add to your local TVM installation permanently

## Making the Fix Permanent (Optional)

If you want to avoid calling the patch every time:

### Option 1: Modify TVM Source Directly

```bash
cd /home/hansz/scratch-data/design/tvm
vim python/tvm/relay/frontend/pytorch.py

# Add the _pad method to PyTorchOpConverter class (around line 2000)
# Add "aten::pad": self._pad to convert_map (around line 3500)

# Restart Python (no rebuild needed for Python-only changes)
```

### Option 2: Create Auto-Import Hook

In your project's `__init__.py`:

```python
# Auto-apply patch when importing tvm-design
from patches.pytorch_pad_support import patch_pytorch_frontend
patch_pytorch_frontend()
```

## Summary

✓ **Quick Fix**: Use the monkey patch (no TVM rebuild needed)  
✓ **Works immediately**: Apply before TVM import  
✓ **Supports all modes**: constant, reflect, replicate, circular  
✓ **Easy to test**: Run provided test scripts  

The patch allows you to compile PyTorch models with padding operations to TVM without modifying TVM source code!

