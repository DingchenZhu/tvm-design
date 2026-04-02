"""
USRNet compiler without any TVM / Relay optimization passes.

Pipeline (only):
  1. Load ONNX via ONNXImporter -> Relay IRModule + params
  2. (Optional) save raw Relay text
  3. analyze_usrnet_structure(mod) -> layer list
  4. generate_usrnet_instructions(layer_info, params) -> backend micro-ops
  5. Write usr_net_no_opt_inst.txt / .bin and params JSON

Explicitly does NOT call PassManager.optimize, relay.transform.*, or any
other optimization pipeline. Relay IR is what the ONNX frontend produces (no
separate TVM optimization pass after import).
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from frontend.onnx_importer import ONNXImporter
from instruction import Inst

from compile_usr_net_with_backend import (
    analyze_usrnet_structure,
    generate_usrnet_instructions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_usr_net_no_opt(model_path: str, output_dir: str) -> dict:
    """
    Import USRNet ONNX to Relay; emit instructions without optimization passes.

    Parameters
    ----------
    model_path : str
        Path to USR_Net.onnx (or compatible ONNX).
    output_dir : str
        Directory for Relay dump, instructions, binary, params metadata.
    """
    logger.info("=" * 60)
    logger.info(
        "USRNet: ONNX -> Relay (NO optimization) -> backend instructions"
    )
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "\n[1/4] Importing ONNX -> Relay (no PassManager / no opt passes)..."
    )
    importer = ONNXImporter()
    mod, params = importer.import_model(model_path)
    logger.info("Imported Relay module; %d parameters", len(params))

    relay_txt = os.path.join(output_dir, "01_imported_relay_no_opt.txt")
    with open(relay_txt, "w", encoding="utf-8") as f:
        f.write(str(mod))
    logger.info("Wrote raw Relay IR to %s", relay_txt)

    logger.info("\n[2/4] Analyzing Relay structure (ExprVisitor on main)...")
    layer_info = analyze_usrnet_structure(mod, params)
    logger.info("Found %d call ops (layer entries)", len(layer_info))
    for i, layer in enumerate(layer_info[:5]):
        logger.info(
            "  Layer %d: %s %s",
            i,
            layer["type"],
            layer.get("shape", ""),
        )

    logger.info(
        "\n[3/4] Generating instructions "
        "(same bridge as compile_usr_net_with_backend)..."
    )
    Inst.code_list = []
    Inst.current_code_num = 0
    instructions = generate_usrnet_instructions(layer_info, params)
    logger.info("Generated %d instructions", len(instructions))

    logger.info("\n[4/4] Packaging...")
    inst_file = os.path.join(output_dir, "usr_net_no_opt_inst.txt")
    with open(inst_file, "w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(str(inst) + "\n")
    logger.info("Wrote %s", inst_file)

    binary_file = os.path.join(output_dir, "usr_net_no_opt_inst.bin")
    try:
        from assembler import compile_file

        compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
        logger.info("Assembled %s", binary_file)
    except Exception as exc:
        logger.error("Assembly failed: %s", exc)

    param_file = os.path.join(output_dir, "usr_net_no_opt_params.json")
    param_info = {}
    for name, param in params.items():
        if hasattr(param, "numpy"):
            param_np = param.numpy()
        elif isinstance(param, np.ndarray):
            param_np = param
        else:
            param_np = np.array(param)
        param_info[name] = {
            "shape": list(param_np.shape),
            "dtype": str(param_np.dtype),
            "size": int(np.prod(param_np.shape)),
        }
    with open(param_file, "w", encoding="utf-8") as f:
        json.dump(param_info, f, indent=2)
    logger.info("Saved %s", param_file)

    return {
        "relay_ir": relay_txt,
        "instructions": inst_file,
        "binary": binary_file,
        "params": param_file,
    }


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(_here, "..", "USR_Net.onnx")
    OUTPUT_DIR = os.path.join(_here, "output", "usr_net_no_opt")

    print("\n" + "=" * 70)
    print("USRNet: no optimization passes — ONNX -> Relay -> instructions")
    print("=" * 70 + "\n")

    if not os.path.isfile(MODEL_PATH):
        logger.error("Model not found: %s", MODEL_PATH)
        sys.exit(1)

    try:
        result = compile_usr_net_no_opt(MODEL_PATH, OUTPUT_DIR)
        print("\nCompilation finished.")
        for key, path in result.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print("  %s: %s (%s bytes)" % (key, path, size))
    except Exception:
        logger.exception("Compilation failed")
        sys.exit(1)
