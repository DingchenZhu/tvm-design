#!/usr/bin/env python3
"""
Additional patch to handle ListType constants in PyTorch frontend.

This fixes the "Unsupported type: ListType" error.
"""

import os
import sys
import shutil

TVM_PYTORCH_FILE = "/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py"
BACKUP_FILE = TVM_PYTORCH_FILE + ".backup2"


def apply_listtype_patch():
    """Apply patch to handle ListType constants."""

    print("=" * 70)
    print("TVM PyTorch Frontend Patch - Adding ListType Support")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(TVM_PYTORCH_FILE):
        print(f"❌ Error: File not found: {TVM_PYTORCH_FILE}")
        return 1

    # Create backup
    print(f"\n📦 Creating backup...")
    shutil.copy2(TVM_PYTORCH_FILE, BACKUP_FILE)
    print(f"   Backup saved to: {BACKUP_FILE}")

    # Read the file
    print(f"\n📖 Reading file...")
    with open(TVM_PYTORCH_FILE, 'r') as f:
        lines = f.readlines()

    # Find the _get_constant function
    print(f"\n🔍 Finding _get_constant function...")

    modified = False
    for i, line in enumerate(lines):
        # Look for the NotImplementedError for unsupported types
        if 'raise NotImplementedError("Unsupported type: %s" % ty)' in line:
            # Check if ListType handling is already added
            if i > 5 and 'ListType' in lines[i-1]:
                print(f"\n✅ ListType handling already exists!")
                return 0

            print(f"   ✓ Found insertion point at line {i+1}")

            # Insert ListType handling before the error
            indent = "    " * 4  # Match the indentation
            listtype_handler = f'''{indent}elif ty == torch.ListType:
{indent}    # Handle list constants (e.g., padding values)
{indent}    val = op_node.t("value")
{indent}    if val.isGenericList():
{indent}        # Convert to Python list
{indent}        list_val = []
{indent}        for item in val.toList():
{indent}            if item.isInt():
{indent}                list_val.append(int(item.toInt()))
{indent}            elif item.isDouble():
{indent}                list_val.append(float(item.toDouble()))
{indent}            else:
{indent}                # Return the raw IValue if we can't convert
{indent}                return val
{indent}        return list_val
{indent}    return val
{indent}'''

            lines.insert(i, listtype_handler)
            modified = True
            break

    if not modified:
        print(f"❌ Error: Could not find insertion point")
        return 1

    # Write back
    print(f"\n💾 Writing patched file...")
    with open(TVM_PYTORCH_FILE, 'w') as f:
        f.writelines(lines)

    print(f"   ✓ File updated successfully")

    print(f"\n✅ Patch applied successfully!")
    print(f"\n📝 Summary:")
    print(f"   - Added ListType constant handling")
    print(f"   - Backup available at: {BACKUP_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(apply_listtype_patch())
