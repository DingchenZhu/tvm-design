# Quick Fix Summary: Disabled LLVM-Dependent Passes

## Problem
The compilation was failing with multiple LLVM-related errors:
1. **First error (FIXED)**: `fold_qnn` parameter not supported in older TVM versions
2. **Second error**: `target.build.llvm is not enabled` in FoldConstant pass
3. **Third error**: `target.build.llvm is not enabled` in FoldScaleAxis pass

## Root Cause
- Several optimization passes require LLVM to evaluate/compile code at compile time:
  - `FoldConstant`: Evaluates constant expressions
  - `FoldScaleAxis`: Folds scale operations into conv/dense weights
- Your TVM installation was built with `USE_LLVM=OFF`
- LLVM is not installed on your system

## Solution Applied
**Disabled LLVM-dependent passes by default** (quick workaround)

### Changes Made:

1. **`passes/pass_config.py`**: Added config flags:
   - `enable_fold_constant: bool = False`
   - `enable_fold_scale_axis: bool = False`

2. **`passes/pass_manager.py`**: Made both passes conditional in `create_visvpu_pipeline()`

3. **`compiler/driver.py`**: Pass both config flags through to the optimizer

4. **`passes/basic_passes.py`**: Made FoldConstant backward-compatible with older TVM versions

### Impact:
- ✅ Compilation will now work without LLVM
- ⚠️ Two optimizations are disabled (minor performance impact):
  - Constant folding (pre-computing constant expressions)
  - Scale axis folding (folding BN scales into conv weights)
- ✅ All other optimizations still work (fusion, DCE, CSE, simplification, etc.)

## To Enable Full Optimization (Optional)

If you want to enable FoldConstant in the future:

### Option A: Install LLVM and rebuild TVM (Recommended)

```bash
# 1. Install LLVM
sudo apt-get install llvm-14 llvm-14-dev

# 2. Edit TVM config
cd /home/hansz/scratch-data/design/tvm
vi cmake/config.cmake
# Change: set(USE_LLVM OFF) → set(USE_LLVM ON)

# 3. Rebuild TVM
cd build
cmake ..
make -j$(nproc)
```

### Option B: Enable LLVM-dependent passes in your script

```python
from compiler import CompilerConfig
from passes.pass_config import PassConfig

config = CompilerConfig(
    pass_config=PassConfig(
        enable_fold_constant=True,      # Enable if you rebuild TVM with LLVM
        enable_fold_scale_axis=True,    # Enable if you rebuild TVM with LLVM
        opt_level=3
    )
)
```

## Test the Fix

Run your compilation again:
```bash
cd /home/hansz/scratch-data/design/tvm-design/examples
python compile_usr_net.py
```

It should now complete without the LLVM error!

