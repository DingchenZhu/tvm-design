#!/usr/bin/env python3
"""
Compare reference (ONNX/PyTorch) output with VPU output to verify functionality.

Use this AFTER you have:
  1. Run the reference model on a test input and saved output (e.g. ref_output.npy)
  2. Run your (or golden) instructions on VPU simulator/hardware with same input
     and saved the VPU output (e.g. vpu_output.npy)

Then run:
  python compare_reference_with_vpu_output.py ref_output.npy vpu_output.npy
  python compare_reference_with_vpu_output.py ref.npy yours.npy golden.npy  # compare all

Exit code 0 if within tolerance, 1 otherwise.
"""

import argparse
import sys
import numpy as np


def load_tensor(path: str) -> np.ndarray:
    """Load a tensor from .npy or .npz (single array)."""
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data
    # .npz: use first array or 'output'
    if 'output' in data.files:
        return data['output']
    return data[data.files[0]]


def compare(a: np.ndarray, b: np.ndarray, rtol: float = 1e-2, atol: float = 1e-2) -> dict:
    """Compare two tensors; return dict of metrics."""
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    if a.size != b.size:
        return {
            "match": False,
            "error": f"Shape mismatch: {a.size} vs {b.size}",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rel_error": None,
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    denom = np.abs(b).astype(np.float64)
    denom[denom < 1e-12] = 1.0
    rel = diff / denom
    rel_err = float(np.max(rel))
    try:
        match = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    except Exception:
        match = False
    return {
        "match": match,
        "error": None,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rel_error": rel_err,
        "atol": atol,
        "rtol": rtol,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare reference and VPU output tensors.")
    ap.add_argument("reference", help="Reference output file (.npy or .npz)")
    ap.add_argument("vpu_outputs", nargs="+", help="VPU output file(s) to compare (e.g. yours.npy golden.npy)")
    ap.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for allclose")
    ap.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for allclose")
    args = ap.parse_args()

    ref = load_tensor(args.reference)
    print(f"Reference shape: {ref.shape} size={ref.size}")
    print()

    all_ok = True
    for path in args.vpu_outputs:
        vpu = load_tensor(path)
        metrics = compare(ref, vpu, rtol=args.rtol, atol=args.atol)
        print(f"  {path}")
        if metrics["error"]:
            print(f"    ERROR: {metrics['error']}")
            all_ok = False
        else:
            print(f"    max_abs_diff = {metrics['max_abs_diff']}")
            print(f"    mean_abs_diff = {metrics['mean_abs_diff']}")
            print(f"    max_rel_error = {metrics['rel_error']}")
            print(f"    within tolerance (rtol={args.rtol}, atol={args.atol}): {'PASS' if metrics['match'] else 'FAIL'}")
            if not metrics["match"]:
                all_ok = False
        print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
