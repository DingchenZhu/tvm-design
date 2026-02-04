#!/usr/bin/env python3
"""
Binary Instruction Decoder

Decodes binary instructions back to human-readable format.
Useful for debugging and verification.
"""

import sys
import ast

sys.path.append('..')
from assembler import (
    FIELD_WIDTHS, OPCODE_MAP, SRC_DEST_WIDTH, OPCODE_WIDTH,
    DATA_LOADER_SRC_BUFFER, DATA_STORER_DEST_BUFFER
)


def reconstruct_instruction_bitstring(binary_lines, start_idx, bit_count):
    """Reconstruct instruction bitstring from 32-bit chunks."""
    bits_needed = bit_count
    bitstring = ""
    chunk_idx = start_idx

    while bits_needed > 0 and chunk_idx < len(binary_lines):
        chunk = binary_lines[chunk_idx]
        if bits_needed >= 32:
            bitstring += chunk
            bits_needed -= 32
        else:
            bitstring += chunk[:bits_needed]
            bits_needed = 0
        chunk_idx += 1

    return bitstring


def decode_opcode(bitstring):
    """Decode opcode from last 3 bits."""
    if len(bitstring) < OPCODE_WIDTH:
        return None, None

    opcode_bits = bitstring[-OPCODE_WIDTH:]
    opcode_val = int(opcode_bits, 2)

    for name, val in OPCODE_MAP.items():
        if val == opcode_val:
            return name, opcode_val

    return f"Unknown({opcode_val})", opcode_val


def decode_instruction(bitstring):
    """Decode a full instruction from bitstring."""
    if len(bitstring) < OPCODE_WIDTH:
        return None

    # Decode opcode (last 3 bits)
    opcode_name, opcode_val = decode_opcode(bitstring)

    if opcode_name not in FIELD_WIDTHS:
        return {
            'op_code': opcode_name,
            'raw_bits': bitstring,
            'error': 'Unknown opcode'
        }

    width_map = FIELD_WIDTHS[opcode_name]

    # Start decoding from the end
    pos = len(bitstring)
    decoded = {'op_code': opcode_name}

    # Opcode (3 bits)
    pos -= OPCODE_WIDTH

    # Dest and src fields (4 bits each)
    for field in ['dest', 'src4', 'src3', 'src2', 'src1']:
        if pos < SRC_DEST_WIDTH:
            break
        pos -= SRC_DEST_WIDTH
        field_bits = bitstring[pos:pos + SRC_DEST_WIDTH]
        decoded[field] = int(field_bits, 2)

    # Decode instruction-specific fields (in reverse order)
    fields_order = list(reversed(width_map.keys()))

    for field in fields_order:
        width = width_map[field]
        if pos < width:
            break
        pos -= width
        field_bits = bitstring[pos:pos + width]
        value = int(field_bits, 2)

        # Map numeric values back to strings for certain fields
        if field == 'src_buffer_idx':
            for name, idx in DATA_LOADER_SRC_BUFFER.items():
                if idx == value:
                    value = name
                    break
        elif field == 'dest_buffer_idx':
            for name, idx in DATA_STORER_DEST_BUFFER.items():
                if idx == value:
                    value = name
                    break

        decoded[field] = value

    decoded['_bits'] = len(bitstring)
    decoded['_chunks'] = (len(bitstring) + 31) // 32

    return decoded


def load_instructions_for_reference(inst_file):
    """Load instructions from text file for comparison."""
    instructions = []
    with open(inst_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                instructions.append(ast.literal_eval(line))
    return instructions


def compare_decoded_with_original(decoded, original):
    """Compare decoded instruction with original."""
    differences = []

    # Compare all fields
    all_keys = set(decoded.keys()) | set(original.keys())
    ignore_keys = {'code_num', 'dependency', 'line_num', '_bits', '_chunks'}

    for key in all_keys:
        if key in ignore_keys or key.startswith('_'):
            continue

        dec_val = decoded.get(key, 'MISSING')
        orig_val = original.get(key, 'MISSING')

        # Handle list values in original
        if isinstance(orig_val, list) and len(orig_val) > 0:
            orig_val = orig_val[0]

        if dec_val != orig_val:
            differences.append(f"  {key}: decoded={dec_val}, original={orig_val}")

    return differences


def main():
    bin_file = './output/usr_net/usr_net_inst.bin'
    inst_file = './output/usr_net/usr_net_inst.txt'

    # Load files
    with open(bin_file, 'r') as f:
        binary_lines = [line.strip() for line in f if line.strip()]

    original_insts = load_instructions_for_reference(inst_file)

    print("=" * 70)
    print("BINARY INSTRUCTION DECODER")
    print("=" * 70)
    print(f"\nBinary chunks: {len(binary_lines)}")
    print(f"Original instructions: {len(original_insts)}")

    # Estimate instruction boundaries
    print("\n" + "=" * 70)
    print("DECODED INSTRUCTIONS (First 5)")
    print("=" * 70)

    chunk_idx = 0
    for inst_num in range(min(5, len(original_insts))):
        orig = original_insts[inst_num]
        opcode = orig.get('op_code')

        # Calculate expected bit count
        if opcode in FIELD_WIDTHS:
            width_map = FIELD_WIDTHS[opcode]
            bit_count = sum(width_map.values()) + 5 * SRC_DEST_WIDTH + OPCODE_WIDTH

            # Reconstruct bitstring
            # Note: This is simplified - actual reconstruction is complex
            # because we don't know exact chunk boundaries
            print(f"\n[{inst_num}] {opcode}")
            print(f"  Expected bits: {bit_count}")
            print(f"  Expected chunks: {(bit_count + 31) // 32}")

            # Show original instruction key fields
            print(f"  Original fields:")
            for key, value in list(orig.items())[:5]:
                if key not in ['code_num', 'dependency', 'line_num']:
                    print(f"    {key}: {value}")

    print("\n" + "=" * 70)
    print("BINARY CHUNK SAMPLES")
    print("=" * 70)

    for i in range(min(10, len(binary_lines))):
        chunk = binary_lines[i]
        ones = chunk.count('1')
        print(f"Chunk {i:3d}: {chunk} (ones: {ones})")

    print("\n" + "=" * 70)
    print("\n📝 Note: Full binary decoding requires tracking exact chunk boundaries")
    print("   The binary format uses 32-bit alignment which adds padding.")
    print("   Use the original instruction file for human-readable format.")


if __name__ == '__main__':
    main()
