#!/usr/bin/env python3
"""
Automatic patch script to add aten::pad support to TVM PyTorch frontend.
"""

import os
import sys
import shutil
from pathlib import Path

# Path to TVM PyTorch frontend
TVM_PYTORCH_FILE = "/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py"

# Backup file
BACKUP_FILE = TVM_PYTORCH_FILE + ".backup"

# Function to add
PAD_GENERIC_FUNCTION = '''
    def pad_generic(self, inputs, input_types):
        """
        Generic pad operator that supports multiple modes.

        PyTorch signature: pad(input, pad, mode='constant', value=0)

        Args:
            inputs[0]: input tensor
            inputs[1]: pad sizes (tuple/list)
            inputs[2]: mode string ('constant', 'reflect', 'replicate', 'circular')
            inputs[3]: value (for constant mode)
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
            # Convert string constant to actual string
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

        # Initialize paddings based on input length
        pad_len = len(self.infer_shape(data)) * 2
        paddings = [0] * pad_len

        # PyTorch pad format: (left, right, top, bottom, front, back)
        # TVM format: [(before, after), (before, after), ...]
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

'''


def apply_patch():
    """Apply the patch to TVM PyTorch frontend."""

    print("=" * 70)
    print("TVM PyTorch Frontend Patch - Adding aten::pad Support")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(TVM_PYTORCH_FILE):
        print(f"❌ Error: TVM PyTorch frontend file not found at:")
        print(f"   {TVM_PYTORCH_FILE}")
        return 1

    # Create backup
    print(f"\n📦 Creating backup...")
    shutil.copy2(TVM_PYTORCH_FILE, BACKUP_FILE)
    print(f"   Backup saved to: {BACKUP_FILE}")

    # Read the file
    print(f"\n📖 Reading PyTorch frontend file...")
    with open(TVM_PYTORCH_FILE, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'def pad_generic(self' in content:
        print(f"\n✅ File is already patched!")
        print(f"   The pad_generic function already exists.")
        return 0

    # Find insertion points
    print(f"\n🔍 Finding insertion points...")

    # 1. Find where to add the function (after make_pad)
    func_marker = "        return pad\n"
    func_insert_idx = content.find(func_marker)

    if func_insert_idx == -1:
        print(f"❌ Error: Could not find function insertion point (make_pad return)")
        return 1

    func_insert_idx += len(func_marker)
    print(f"   ✓ Found function insertion point at position {func_insert_idx}")

    # 2. Find where to add the operator mapping
    op_marker = '"aten::replication_pad3d": self.make_pad("edge"),'
    op_insert_idx = content.find(op_marker)

    if op_insert_idx == -1:
        print(f"❌ Error: Could not find operator mapping insertion point")
        return 1

    op_insert_idx = content.find('\n', op_insert_idx) + 1
    print(f"   ✓ Found operator mapping insertion point at position {op_insert_idx}")

    # Apply patches
    print(f"\n✏️  Applying patches...")

    # Insert function
    new_content = content[:func_insert_idx] + PAD_GENERIC_FUNCTION + content[func_insert_idx:]

    # Recalculate op_insert_idx after adding function
    op_insert_idx += len(PAD_GENERIC_FUNCTION)

    # Insert operator mapping
    op_mapping = '            "aten::pad": self.pad_generic,\n'
    new_content = new_content[:op_insert_idx] + op_mapping + new_content[op_insert_idx:]

    print(f"   ✓ Added pad_generic function")
    print(f"   ✓ Added aten::pad operator mapping")

    # Write back
    print(f"\n💾 Writing patched file...")
    with open(TVM_PYTORCH_FILE, 'w') as f:
        f.write(new_content)

    print(f"   ✓ File updated successfully")

    # Verify
    print(f"\n✅ Patch applied successfully!")
    print(f"\n📝 Summary:")
    print(f"   - Added pad_generic() function")
    print(f"   - Registered aten::pad operator")
    print(f"   - Backup available at: {BACKUP_FILE}")

    print(f"\n🧪 Next steps:")
    print(f"   1. Test the patch: python compile_fsrcnn.py")
    print(f"   2. If issues occur, restore backup:")
    print(f"      cp {BACKUP_FILE} {TVM_PYTORCH_FILE}")

    return 0


def restore_backup():
    """Restore the backup file."""
    if not os.path.exists(BACKUP_FILE):
        print(f"❌ Error: Backup file not found at {BACKUP_FILE}")
        return 1

    shutil.copy2(BACKUP_FILE, TVM_PYTORCH_FILE)
    print(f"✅ Restored backup from {BACKUP_FILE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        sys.exit(restore_backup())
    else:
        sys.exit(apply_patch())
