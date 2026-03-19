#!/usr/bin/env python3
"""
Compare your compiler's instruction output with golden reference.

How to check if your function is the same as the golden
========================================================

1. Structural comparison (this script)
   - Run: python compare_with_golden.py [--mine your_inst.txt] [--golden golden.txt]
   - Compares logical op sequence after normalizing and (optionally) collapsing
     the golden's per-row unrolled instructions.
   - If golden is for a *different* model (e.g. SD UNet vs USRNet), counts and
     sequence will differ; use --summary-only to compare op/layer mix only.

2. Full functional equivalence (same outputs for same inputs)
   - Run reference model (ONNX/PyTorch) on test input -> save reference output.
   - Run your instruction stream on VPU simulator or hardware with same input.
   - Compare reference output vs VPU output (e.g. max |diff|, correlation).
   - This script does not run a simulator; it only checks structure.

Usage:
  python compare_with_golden.py [--mine path] [--golden path] [--golden-dir dir]
  python compare_with_golden.py --summary-only   # op/layer counts only
  python compare_with_golden.py                  # defaults: output/usr_net/usr_net_inst.txt, golden/
"""

import argparse
import ast
import os
import sys
from collections import defaultdict

# Default paths relative to script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MINE = os.path.join(SCRIPT_DIR, "output", "usr_net", "usr_net_inst.txt")
DEFAULT_GOLDEN_DIR = os.path.join(SCRIPT_DIR, "..", "golden")


# Fields that define "logical" behavior (same value = same operation semantics)
COMPARE_FIELDS = {
    "op_code",
    "layer_idx",
    "transnum",
    "stride",
    "acc_mode",
    "transfer_num",
    "store_mode",
    "line_buffer_idx",
    "reg_out_idx",
    "quant_mode",
    "quant_reg_load_idx",
    "read_mode",
    "is_padding_row",
    "dest_buffer_idx",
    "src_buffer_idx",
    "load_model",
}

# Fields to ignore when comparing (scheduling / per-row addresses)
IGNORE_FIELDS = {
    "code_num",
    "dependency",
    "dest",
    "src1",
    "src2",
    "src3",
    "src4",
    "base_addrs_res",
    "base_addr_pooling",
    "bas_addr",
    "is_compression",
    "offchip_read_mode",
    "is_offset",
    "is_skip",
    "is_new",
}


def parse_line(line: str):
    """Parse one line; handle optional 'index: {...}' prefix."""
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


def load_instructions(path: str):
    """Load list of instruction dicts from a text file (one dict per line)."""
    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    instructions = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            inst = parse_line(line)
            if inst is not None and isinstance(inst, dict):
                inst["_line"] = line_num
                instructions.append(inst)
    return instructions, None


def logical_key(inst: dict) -> tuple:
    """Build a comparable key from an instruction (for grouping and comparison)."""
    key_parts = [inst.get("op_code", "")]
    for f in sorted(COMPARE_FIELDS):
        if f not in inst:
            continue
        v = inst[f]
        if isinstance(v, list):
            v = v[0] if v else 0
        key_parts.append((f, v))
    return tuple(key_parts)


def normalize_inst(inst: dict) -> dict:
    """Return a minimal dict with only comparison-relevant fields."""
    out = {}
    for f in COMPARE_FIELDS:
        if f in inst:
            v = inst[f]
            if isinstance(v, list) and len(v) > 0:
                v = v[0]
            out[f] = v
    if "op_code" not in out and "op_code" in inst:
        out["op_code"] = inst["op_code"]
    return out


def collapse_unrolled(instructions: list) -> list:
    """
    Collapse consecutive instructions that are the same logical op
    (e.g. golden's per-row unrolling -> one logical op per layer block).
    """
    if not instructions:
        return []
    out = []
    prev_key = None
    for inst in instructions:
        k = logical_key(inst)
        if k != prev_key:
            out.append(normalize_inst(inst))
            prev_key = k
    return out


def compare_sequences(mine_norm: list, golden_collapsed: list) -> tuple:
    """
    Compare two sequences of normalized instructions.
    Returns (match: bool, list of diff messages).
    """
    diffs = []
    n_mine = len(mine_norm)
    n_golden = len(golden_collapsed)
    if n_mine != n_golden:
        diffs.append(f"Instruction count: yours={n_mine}, golden (collapsed)={n_golden}")
    for i in range(min(n_mine, n_golden)):
        a, b = mine_norm[i], golden_collapsed[i]
        for k in set(a) | set(b):
            if k in IGNORE_FIELDS or k.startswith("_"):
                continue
            va, vb = a.get(k), b.get(k)
            if va != vb:
                diffs.append(f"  [{i}] {k}: yours={va}, golden={vb}")
    if len(diffs) == 0 and n_mine == n_golden:
        return True, []
    return len(diffs) == 0, diffs


def summary_stats(instructions: list) -> dict:
    """Per op_code and layer_idx counts."""
    by_op = defaultdict(int)
    by_layer = defaultdict(lambda: defaultdict(int))
    for inst in instructions:
        op = inst.get("op_code", "?")
        by_op[op] += 1
        layer = inst.get("layer_idx", "?")
        by_layer[layer][op] += 1
    return {"by_op": dict(by_op), "by_layer": {k: dict(v) for k, v in by_layer.items()}}


def main():
    ap = argparse.ArgumentParser(description="Compare your compiler output with golden.")
    ap.add_argument("--mine", default=DEFAULT_MINE, help="Your instruction .txt file")
    ap.add_argument("--golden", default=None, help="Single golden .txt file")
    ap.add_argument(
        "--golden-dir",
        default=DEFAULT_GOLDEN_DIR,
        help="Directory with golden pseudo_code_load_next_*.txt (used if --golden not set)",
    )
    ap.add_argument(
        "--segment",
        choices=["first", "mid", "last"],
        default="first",
        help="Which golden segment to use when using --golden-dir",
    )
    ap.add_argument(
        "--collapse-golden",
        action="store_true",
        default=True,
        help="Collapse golden unrolled instructions to logical ops (default: True)",
    )
    ap.add_argument(
        "--no-collapse",
        action="store_false",
        dest="collapse_golden",
        help="Do not collapse golden; compare raw (will only match if you also unroll)",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary stats (op/layer counts), no sequence comparison",
    )
    args = ap.parse_args()

    # Resolve paths
    mine_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.mine) if not os.path.isabs(args.mine) else args.mine)
    if args.golden:
        golden_path = os.path.normpath(
            os.path.join(SCRIPT_DIR, args.golden) if not os.path.isabs(args.golden) else args.golden
        )
    else:
        golden_dir = os.path.normpath(
            os.path.join(SCRIPT_DIR, args.golden_dir) if not os.path.isabs(args.golden_dir) else args.golden_dir
        )
        golden_path = os.path.join(golden_dir, f"pseudo_code_load_next_{args.segment}.txt")

    print("=" * 60)
    print("Compare with golden")
    print("=" * 60)
    print(f"  Mine:   {mine_path}")
    print(f"  Golden: {golden_path}")
    print()

    # Load
    mine_inst, err = load_instructions(mine_path)
    if err or mine_inst is None:
        print(f"[ERROR] {err or 'Failed to load mine'}")
        sys.exit(1)
    golden_inst, err = load_instructions(golden_path)
    if err or golden_inst is None:
        print(f"[ERROR] Golden: {err or 'Failed to load golden'}")
        sys.exit(1)

    print(f"Loaded: yours={len(mine_inst)} instructions, golden={len(golden_inst)} instructions")
    if args.collapse_golden:
        golden_collapsed = collapse_unrolled(golden_inst)
        print(f"Golden collapsed to {len(golden_collapsed)} logical instructions")
    else:
        golden_collapsed = [normalize_inst(i) for i in golden_inst]
    mine_norm = [normalize_inst(i) for i in mine_inst]
    print()

    if args.summary_only:
        print("--- Yours (summary) ---")
        for op, cnt in sorted(summary_stats(mine_inst)["by_op"].items()):
            print(f"  {op}: {cnt}")
        print("--- Golden (summary, after collapse) ---")
        for op, cnt in sorted(summary_stats(golden_collapsed)["by_op"].items()):
            print(f"  {op}: {cnt}")
        return

    # Compare
    match, diffs = compare_sequences(mine_norm, golden_collapsed)
    if match:
        print("[PASS] Logical instruction sequence matches golden (after normalizing/collapsing).")
    else:
        print("[DIFF] Sequences differ.")
        for d in diffs[:50]:
            print(d)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs) - 50} more differences")
    print()
    print("Note: Full functional equivalence (same outputs for same inputs) requires:")
    print("  1. Run reference model (ONNX/PyTorch) on test input -> reference output")
    print("  2. Run your instructions on VPU simulator/hardware with same input -> your output")
    print("  3. Compare reference output vs your output (e.g. max abs diff, correlation)")
    print("This script only checks structural / logical-op equivalence.")


if __name__ == "__main__":
    main()
