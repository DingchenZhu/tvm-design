#!/usr/bin/env python3
"""
Binary Format Analyzer

Analyzes the binary instruction format and provides detailed statistics.
"""

import sys
import ast

sys.path.append('..')
from assembler import FIELD_WIDTHS, OPCODE_MAP, SRC_DEST_WIDTH, OPCODE_WIDTH


def load_binary(bin_file):
    """Load binary file."""
    with open(bin_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


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


def decode_opcode(bitstr):
    """Decode opcode from last 3 bits."""
    opcode_bits = bitstr[-OPCODE_WIDTH:]
    opcode_val = int(opcode_bits, 2)

    # Find opcode name
    for name, val in OPCODE_MAP.items():
        if val == opcode_val:
            return name, opcode_val

    return f"Unknown({opcode_val})", opcode_val


def analyze_instruction_encoding(instructions):
    """Analyze how instructions are encoded."""
    encoding_stats = {
        'bits_by_opcode': {},
        'total_bits': 0,
    }

    for inst in instructions:
        op_code = inst.get('op_code')
        if not op_code or op_code not in FIELD_WIDTHS:
            continue

        width_map = FIELD_WIDTHS[op_code]

        # Calculate total bits for this instruction
        # Fields from width_map
        inst_bits = sum(width_map.values())
        # Plus src1-4, dest (5 * 4 bits each)
        inst_bits += 5 * SRC_DEST_WIDTH
        # Plus opcode (3 bits)
        inst_bits += OPCODE_WIDTH

        if op_code not in encoding_stats['bits_by_opcode']:
            encoding_stats['bits_by_opcode'][op_code] = {
                'min': inst_bits,
                'max': inst_bits,
                'total': 0,
                'count': 0
            }

        stats = encoding_stats['bits_by_opcode'][op_code]
        stats['min'] = min(stats['min'], inst_bits)
        stats['max'] = max(stats['max'], inst_bits)
        stats['total'] += inst_bits
        stats['count'] += 1
        encoding_stats['total_bits'] += inst_bits

    return encoding_stats


def analyze_binary_chunks(binary_lines):
    """Analyze binary chunk distribution."""
    stats = {
        'total_chunks': len(binary_lines),
        'bits_per_chunk': 32,
        'total_bits': len(binary_lines) * 32,
        'zero_chunks': 0,
        'ones_distribution': [],
    }

    for line in binary_lines:
        ones = line.count('1')
        stats['ones_distribution'].append(ones)
        if ones == 0:
            stats['zero_chunks'] += 1

    return stats


def print_analysis(instructions, binary_lines):
    """Print comprehensive analysis."""
    print("=" * 70)
    print("BINARY FORMAT ANALYSIS")
    print("=" * 70)

    # Basic info
    print("\n📊 BASIC STATISTICS")
    print("-" * 70)
    print(f"Instructions:           {len(instructions)}")
    print(f"Binary chunks (32-bit): {len(binary_lines)}")
    print(f"Chunks per instruction: {len(binary_lines) / len(instructions):.2f}")
    print(f"Total binary size:      {len(binary_lines) * 32} bits ({len(binary_lines) * 4} bytes)")

    # Encoding analysis
    print("\n🔧 INSTRUCTION ENCODING")
    print("-" * 70)
    encoding_stats = analyze_instruction_encoding(instructions)

    print(f"Total instruction bits: {encoding_stats['total_bits']}")
    print(f"Average bits/inst:      {encoding_stats['total_bits'] / len(instructions):.1f}")
    print()
    print("Bits per opcode:")
    for op_code, stats in sorted(encoding_stats['bits_by_opcode'].items()):
        avg = stats['total'] / stats['count']
        print(f"  {op_code:20s}: {avg:.0f} bits ({stats['count']} insts)")

    # Calculate padding
    total_inst_bits = encoding_stats['total_bits']
    total_binary_bits = len(binary_lines) * 32
    padding_bits = total_binary_bits - total_inst_bits
    padding_percent = (padding_bits / total_binary_bits) * 100

    print(f"\n📦 PADDING ANALYSIS")
    print("-" * 70)
    print(f"Instruction bits:       {total_inst_bits}")
    print(f"Binary file bits:       {total_binary_bits}")
    print(f"Padding bits:           {padding_bits} ({padding_percent:.1f}%)")
    print(f"Padding reason:         32-bit alignment for hardware")

    # Binary chunk analysis
    print("\n💾 BINARY CHUNK ANALYSIS")
    print("-" * 70)
    chunk_stats = analyze_binary_chunks(binary_lines)

    print(f"Total 32-bit chunks:    {chunk_stats['total_chunks']}")
    print(f"Zero chunks:            {chunk_stats['zero_chunks']}")
    print(f"Non-zero chunks:        {chunk_stats['total_chunks'] - chunk_stats['zero_chunks']}")

    ones_dist = chunk_stats['ones_distribution']
    print(f"Avg ones per chunk:     {sum(ones_dist) / len(ones_dist):.1f}")
    print(f"Min ones per chunk:     {min(ones_dist)}")
    print(f"Max ones per chunk:     {max(ones_dist)}")

    # Sample decoded instructions
    print("\n🔍 SAMPLE DECODED INSTRUCTIONS")
    print("-" * 70)

    # Show first 3 instructions
    for i in range(min(3, len(instructions))):
        inst = instructions[i]
        print(f"\nInstruction {i} (Line {inst.get('line_num', '?')}):")
        print(f"  Opcode:     {inst.get('op_code', 'N/A')}")

        # Show key fields
        op_code = inst.get('op_code')
        if op_code in FIELD_WIDTHS:
            print(f"  Fields:")
            width_map = FIELD_WIDTHS[op_code]
            for field, width in list(width_map.items())[:5]:  # Show first 5 fields
                value = inst.get(field, 'N/A')
                print(f"    {field:20s}: {value} ({width} bits)")
            if len(width_map) > 5:
                print(f"    ... and {len(width_map) - 5} more fields")

    # Validation summary
    print("\n" + "=" * 70)
    print("✅ VALIDATION SUMMARY")
    print("=" * 70)

    checks = [
        ("Instructions loaded", len(instructions) > 0),
        ("Binary chunks generated", len(binary_lines) > 0),
        ("32-bit alignment", all(len(line) == 32 for line in binary_lines)),
        ("Binary format valid", all(all(c in '01' for c in line) for line in binary_lines)),
        ("Chunk ratio reasonable", 1.5 <= len(binary_lines) / len(instructions) <= 4.0),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:30s}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Binary format is correct!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues above")
    print("=" * 70)


def main():
    inst_file = './output/usr_net/usr_net_inst.txt'
    bin_file = './output/usr_net/usr_net_inst.bin'

    print("Loading files...")
    instructions = load_instructions(inst_file)
    binary_lines = load_binary(bin_file)

    print(f"Loaded {len(instructions)} instructions")
    print(f"Loaded {len(binary_lines)} binary chunks\n")

    print_analysis(instructions, binary_lines)


if __name__ == '__main__':
    main()
