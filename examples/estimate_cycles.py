#!/usr/bin/env python3
"""
Estimate total cycle count for an instruction stream (and optionally compare with golden).

Uses a simple per-op cycle model (see CYCLES_PER_OP below). This is NOT real hardware
timing—only for relative comparison (e.g. your stream vs golden) when you don't have
a simulator. Real performance depends on pipeline, parallelism, and memory.

Usage:
  python estimate_cycles.py inst.txt
  python estimate_cycles.py mine.txt --golden ../golden/pseudo_code_load_next_first.txt
"""

import argparse
import ast
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default cycles per instruction (from VERIFICATION_REPORT.md / typical VPU assumptions).
# For ops with transnum/transfer_num, we use fixed cycle per inst for simplicity;
# you can change to scale by transnum if your model does.
CYCLES_PER_OP = {
    "OffchipDataLoader": 100,
    "OffchipDataStorer": 50,
    "QuantLoader": 4,
    "DataLoader": 15,
    "WeightLoader": 9,
    "DataStorer": 32,
    "OffsetLoader": 8,
}


def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    if line[0].isdigit():
        i = 0
        while i < len(line) and line[i].isdigit():
            i += 1
        if i < len(line) and line[i] == ":":
            line = line[i + 1 :].strip()
    try:
        return ast.literal_eval(line)
    except Exception:
        return None


def load_instructions(path: str) -> list:
    if not os.path.isfile(path):
        return None
    out = []
    with open(path, "r") as f:
        for line in f:
            inst = parse_line(line)
            if inst is not None and isinstance(inst, dict) and inst.get("op_code"):
                out.append(inst)
    return out


def estimate_cycles(instructions: list) -> tuple:
    """Return (total_cycles, per_op_counts, per_op_cycles)."""
    total = 0
    per_op_count = {}
    per_op_cycles = {}
    for inst in instructions:
        op = inst.get("op_code", "?")
        cy = CYCLES_PER_OP.get(op, 10)
        total += cy
        per_op_count[op] = per_op_count.get(op, 0) + 1
        per_op_cycles[op] = per_op_cycles.get(op, 0) + cy
    return total, per_op_count, per_op_cycles


def main():
    ap = argparse.ArgumentParser(description="Estimate cycles for instruction stream(s).")
    ap.add_argument("inst_file", help="Your instruction .txt file")
    ap.add_argument("--golden", help="Golden instruction .txt for comparison")
    ap.add_argument("--no-golden", action="store_true", help="Only print estimate for inst_file")
    args = ap.parse_args()

    path = args.inst_file if os.path.isabs(args.inst_file) else os.path.join(SCRIPT_DIR, args.inst_file)
    insts = load_instructions(path)
    if insts is None:
        print(f"Failed to load: {path}", file=sys.stderr)
        sys.exit(1)

    total, per_op_count, per_op_cycles = estimate_cycles(insts)
    print("=" * 60)
    print("Cycle estimate (simple model; not real hardware timing)")
    print("=" * 60)
    print(f"  File: {path}")
    print(f"  Instructions: {len(insts)}")
    print(f"  Estimated total cycles: {total}")
    print()
    print("  Per op:")
    for op in sorted(per_op_count.keys()):
        print(f"    {op}: count={per_op_count[op]}, cycles={per_op_cycles[op]}")
    print()

    if args.golden and not args.no_golden:
        gpath = args.golden if os.path.isabs(args.golden) else os.path.join(SCRIPT_DIR, args.golden)
        ginsts = load_instructions(gpath)
        if ginsts is None:
            print(f"Golden not loaded: {gpath}", file=sys.stderr)
        else:
            gtotal, g_per_op_count, g_per_op_cycles = estimate_cycles(ginsts)
            print("  Golden:")
            print(f"    File: {gpath}")
            print(f"    Instructions: {len(ginsts)}")
            print(f"    Estimated total cycles: {gtotal}")
            print()
            print("  Comparison:")
            print(f"    Your estimated cycles:   {total}")
            print(f"    Golden estimated cycles: {gtotal}")
            if gtotal > 0:
                ratio = total / gtotal
                print(f"    Ratio (yours/golden):    {ratio:.3f}")
            print()
            print("  Note: Golden is unrolled (many more instructions); real hardware")
            print("  may overlap loads/stores. Use simulator/hardware for real performance.")

    print()
    print("  For real performance: run on VPU simulator or hardware and measure cycles.")


if __name__ == "__main__":
    main()
