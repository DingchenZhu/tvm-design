#!/usr/bin/env python3
"""
Instruction Verification Tool

This script verifies:
1. Instruction format and field correctness
2. Bit-width constraints
3. Binary encoding correctness
4. Instruction sequence patterns
"""

import sys
import ast
from collections import defaultdict, Counter

# Import assembler configuration
sys.path.append('..')
from assembler import FIELD_WIDTHS, OPCODE_MAP, SRC_DEST_WIDTH, OPCODE_WIDTH

def load_instructions(inst_file):
    """Load instructions from text file."""
    instructions = []
    with open(inst_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                inst = ast.literal_eval(line)
                inst['line_num'] = line_num
                instructions.append(inst)
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to parse - {e}")
    return instructions


def verify_bit_widths(instructions):
    """Verify all field values fit within their bit-width constraints."""
    errors = []
    warnings = []

    for inst in instructions:
        op_code = inst.get('op_code')
        if not op_code:
            errors.append(f"Line {inst['line_num']}: Missing op_code")
            continue

        if op_code not in FIELD_WIDTHS:
            errors.append(f"Line {inst['line_num']}: Unknown op_code '{op_code}'")
            continue

        width_map = FIELD_WIDTHS[op_code]

        # Check each field
        for field, width in width_map.items():
            if field not in inst:
                continue

            value = inst[field]

            # Handle string values (buffer indices)
            if isinstance(value, str):
                continue

            # Handle list values
            if isinstance(value, list):
                value = value[0] if value else 0

            # Check if value fits in bit width
            max_val = (1 << width) - 1
            if value < 0:
                errors.append(f"Line {inst['line_num']}: {op_code}.{field} = {value} (negative value not allowed)")
            elif value > max_val:
                errors.append(f"Line {inst['line_num']}: {op_code}.{field} = {value} exceeds {width}-bit max ({max_val})")

        # Check src/dest fields
        for field in ['src1', 'src2', 'src3', 'src4', 'dest']:
            if field in inst:
                value = inst[field]
                if isinstance(value, list):
                    value = value[0] if value else 0
                max_val = (1 << SRC_DEST_WIDTH) - 1
                if value > max_val:
                    errors.append(f"Line {inst['line_num']}: {field} = {value} exceeds {SRC_DEST_WIDTH}-bit max ({max_val})")

    return errors, warnings


def analyze_instruction_patterns(instructions):
    """Analyze instruction sequences and patterns."""
    stats = {
        'total': len(instructions),
        'by_opcode': Counter(),
        'layer_indices': defaultdict(list),
        'buffer_usage': defaultdict(int),
    }

    for inst in instructions:
        op_code = inst.get('op_code')
        stats['by_opcode'][op_code] += 1

        # Track layer indices
        if 'layer_idx' in inst:
            stats['layer_indices'][inst['layer_idx']].append(inst['line_num'])

        # Track buffer usage
        if 'src_buffer_idx' in inst:
            stats['buffer_usage'][f"src:{inst['src_buffer_idx']}"] += 1
        if 'dest_buffer_idx' in inst:
            stats['buffer_usage'][f"dest:{inst['dest_buffer_idx']}"] += 1

    return stats


def check_instruction_sequence(instructions):
    """Check for common instruction sequence patterns."""
    issues = []

    # Check for standard conv pattern: QuantLoader -> DataLoader -> WeightLoader -> DataStorer
    for i in range(len(instructions) - 3):
        seq = [instructions[i+j].get('op_code') for j in range(4)]

        # Expected pattern for conv layer
        if seq[0] == 'QuantLoader' and seq[1] == 'DataLoader':
            if seq[2] != 'WeightLoader':
                issues.append(f"Line {instructions[i]['line_num']}: Unexpected sequence after DataLoader (expected WeightLoader, got {seq[2]})")
            if seq[3] != 'DataStorer':
                issues.append(f"Line {instructions[i]['line_num']}: Unexpected sequence after WeightLoader (expected DataStorer, got {seq[3]})")

    return issues


def verify_binary_format(bin_file, instructions):
    """Verify binary file format."""
    with open(bin_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    errors = []

    if len(lines) != len(instructions):
        errors.append(f"Binary line count ({len(lines)}) doesn't match instruction count ({len(instructions)})")

    for i, line in enumerate(lines[:10], 1):  # Check first 10 lines
        if not all(c in '01' for c in line):
            errors.append(f"Binary line {i}: Contains non-binary characters")

    return errors


def print_report(instructions, errors, warnings, stats, seq_issues, bin_errors):
    """Print verification report."""
    print("=" * 70)
    print("INSTRUCTION VERIFICATION REPORT")
    print("=" * 70)

    # Summary
    print("\n📊 SUMMARY")
    print("-" * 70)
    print(f"Total Instructions:      {stats['total']}")
    print(f"Critical Errors:         {len(errors)}")
    print(f"Warnings:               {len(warnings)}")
    print(f"Sequence Issues:        {len(seq_issues)}")
    print(f"Binary Errors:          {len(bin_errors)}")

    # Instruction breakdown
    print("\n📋 INSTRUCTION BREAKDOWN")
    print("-" * 70)
    for op_code, count in sorted(stats['by_opcode'].items(), key=lambda x: -x[1]):
        print(f"  {op_code:20s}: {count:4d} ({count/stats['total']*100:5.1f}%)")

    # Buffer usage
    print("\n💾 BUFFER USAGE")
    print("-" * 70)
    for buf, count in sorted(stats['buffer_usage'].items()):
        print(f"  {buf:25s}: {count:4d}")

    # Layer index distribution
    print("\n🔢 LAYER INDEX DISTRIBUTION")
    print("-" * 70)
    layer_counts = {idx: len(lines) for idx, lines in stats['layer_indices'].items()}
    for layer_idx in sorted(layer_counts.keys())[:10]:  # Show first 10
        print(f"  Layer {layer_idx:2d}: {layer_counts[layer_idx]:3d} instructions")
    if len(layer_counts) > 10:
        print(f"  ... and {len(layer_counts) - 10} more layers")

    # Errors
    if errors:
        print("\n❌ CRITICAL ERRORS")
        print("-" * 70)
        for error in errors[:20]:  # Show first 20
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    # Warnings
    if warnings:
        print("\n⚠️  WARNINGS")
        print("-" * 70)
        for warning in warnings[:10]:
            print(f"  {warning}")

    # Sequence issues
    if seq_issues:
        print("\n⚡ SEQUENCE ISSUES")
        print("-" * 70)
        for issue in seq_issues[:10]:
            print(f"  {issue}")

    # Binary errors
    if bin_errors:
        print("\n🔧 BINARY FORMAT ERRORS")
        print("-" * 70)
        for error in bin_errors:
            print(f"  {error}")

    # Overall status
    print("\n" + "=" * 70)
    if not errors and not bin_errors:
        print("✅ VERIFICATION PASSED - All checks successful!")
    elif not errors:
        print("⚠️  VERIFICATION PASSED WITH WARNINGS")
    else:
        print("❌ VERIFICATION FAILED - Please fix errors above")
    print("=" * 70)

    return len(errors) == 0 and len(bin_errors) == 0


def main():
    inst_file = './output/usr_net/usr_net_inst.txt'
    bin_file = './output/usr_net/usr_net_inst.bin'

    print("Loading instructions...")
    instructions = load_instructions(inst_file)
    print(f"Loaded {len(instructions)} instructions\n")

    print("Verifying bit-width constraints...")
    errors, warnings = verify_bit_widths(instructions)

    print("Analyzing instruction patterns...")
    stats = analyze_instruction_patterns(instructions)

    print("Checking instruction sequences...")
    seq_issues = check_instruction_sequence(instructions)

    print("Verifying binary format...")
    bin_errors = verify_binary_format(bin_file, instructions)

    print("\n")
    success = print_report(instructions, errors, warnings, stats, seq_issues, bin_errors)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
