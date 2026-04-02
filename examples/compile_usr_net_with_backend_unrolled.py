"""
USRNet compilation with UNROLLED instructions (one instruction block per output row pair).

Same pipeline as compile_usr_net_with_backend.py, but conv and deformable_conv layers
are unrolled: for each output row (or pair of rows) we emit a full set of
DataLoader/WeightLoader/DataStorer with incremented base_addrs_res and bas_addr,
so the output is a linear instruction stream that does not require hardware loop support.

Output: output/usr_net_unroll/usr_net_unroll_inst.txt, usr_net_unroll_inst.bin, etc.

Why ~7k instructions vs golden's 17k+?
- Rows per layer: USRNet layers use output height from shape (often 32); golden uses
  144 rows per layer (stride 144). So golden has 144/32 ≈ 4.5x more instructions per
  conv layer. Set UNROLL_OUTPUT_ROWS_OVERRIDE = 144 in this file to match that scale.
- Different model: Golden is a different pipeline (e.g. SD UNet, 19 layers, 144 rows);
  USRNet has different layer count and spatial sizes, so total instruction count differs.
"""

import sys
import os
sys.path.append('..')

import logging
import numpy as np
import json

from frontend.onnx_importer import ONNXImporter
from passes.pass_manager import PassManager
from instruction import Inst

# Reuse analysis and non-unrolled layer handlers from the original compiler
from compile_usr_net_with_backend import (
    analyze_usrnet_structure,
    _generate_activation_instructions,
    _generate_elementwise_instructions,
    _generate_norm_instructions,
    _generate_pooling_instructions,
    _generate_concat_instructions,
    _generate_upsample_instructions,
    _generate_clip_instructions,
    _generate_layout_transform_instructions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default number of output rows per layer when shape is unknown (e.g. 32x32 tile)
DEFAULT_OUTPUT_ROWS = 32
# Override: set to a positive int to force this many rows for every conv/deformable layer
# (e.g. 144 to match golden's scale; golden uses 144 rows and stride 144, so ~17k instructions)
UNROLL_OUTPUT_ROWS_OVERRIDE = None  # or 144 to match golden row count
STRIDE = 32


def _get_output_rows(layer: dict) -> int:
    """Infer number of output rows from layer shape [N, C, H, W]. Default DEFAULT_OUTPUT_ROWS."""
    if UNROLL_OUTPUT_ROWS_OVERRIDE is not None and UNROLL_OUTPUT_ROWS_OVERRIDE > 0:
        return int(UNROLL_OUTPUT_ROWS_OVERRIDE)
    shape = layer.get('shape')
    if isinstance(shape, list) and len(shape) >= 3:
        return int(shape[2])  # H
    return DEFAULT_OUTPUT_ROWS


def _generate_conv_instructions_unrolled(layer_idx: int, layer: dict):
    """Generate UNROLLED instructions for standard convolution (one block per pair of output rows)."""
    from instruction import DataLoader, WeightLoader, QuantLoader, DataStorer

    hw_layer_idx = layer_idx % 32
    num_rows = _get_output_rows(layer)
    num_blocks = (num_rows + 1) // 2  # 2 rows per block (double line buffer)

    # One QuantLoader per layer
    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0,
    )

    for block in range(num_blocks):
        base_addr_0 = block * 4
        base_addr_1 = block * 4 + 2
        base_addrs_res_0 = block * 2
        base_addrs_res_1 = block * 2 + 1
        is_first_block = block == 0

        # DataLoader line 0
        DataLoader.dispatch(
            layer_idx=hw_layer_idx,
            line_buffer_reshape=0,
            is_padding_row=1 if is_first_block else 0,
            read_mode=0,
            transnum=15,
            line_buffer_idx=0,
            src_buffer_idx='a',
            bas_addr=base_addr_0,
        )
        # DataLoader line 1
        DataLoader.dispatch(
            layer_idx=hw_layer_idx,
            line_buffer_reshape=0,
            is_padding_row=0,
            read_mode=0,
            transnum=15,
            line_buffer_idx=1,
            src_buffer_idx='b',
            bas_addr=base_addr_1,
        )
        # WeightLoader acc 0
        WeightLoader.dispatch(
            acc_reg_comp_idx=0,
            kernal_size=0,
            line_buffer_row_shift=1,
            line_buffer_idx=0,
            is_padding_col=1,
            weight_parall_mode=0,
            is_new=1 if is_first_block else 0,
            transnum=9,
            bas_addr=0,
            is_bilinear_bicubic=0,
            offset_reg_idx=0,
        )
        # WeightLoader acc 1
        WeightLoader.dispatch(
            acc_reg_comp_idx=1,
            kernal_size=0,
            line_buffer_row_shift=1,
            line_buffer_idx=1,
            is_padding_col=1,
            weight_parall_mode=0,
            is_new=1 if is_first_block else 0,
            transnum=9,
            bas_addr=0,
            is_bilinear_bicubic=0,
            offset_reg_idx=0,
        )
        # DataStorer reg 0
        DataStorer.dispatch(
            quant_config_idx=0,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=0,
            acc_mode=0,
            transfer_num=1,
            store_mode=0,
            stride=STRIDE,
            base_addr_pooling=0,
            base_addrs_res=base_addrs_res_0,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx='b',
        )
        # DataStorer reg 1
        DataStorer.dispatch(
            quant_config_idx=0,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=1,
            acc_mode=0,
            transfer_num=1,
            store_mode=0,
            stride=STRIDE,
            base_addr_pooling=0,
            base_addrs_res=base_addrs_res_1,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx='b',
        )


def _generate_deformable_conv_instructions_unrolled(layer_idx: int, layer: dict):
    """Generate UNROLLED instructions for deformable convolution."""
    from instruction import (
        DataLoader, WeightLoader, OffsetLoader,
        QuantLoader, DataStorer,
    )

    hw_layer_idx = layer_idx % 32
    num_rows = _get_output_rows(layer)
    num_blocks = (num_rows + 1) // 2

    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0,
    )
    OffsetLoader.dispatch(offset_reg_idx=0, bas_addr=0)

    for block in range(num_blocks):
        base_addr_0 = block * 4
        base_addr_1 = block * 4 + 2
        base_addrs_res_0 = block * 2
        base_addrs_res_1 = block * 2 + 1
        is_first_block = block == 0

        DataLoader.dispatch(
            layer_idx=hw_layer_idx,
            line_buffer_reshape=0,
            is_padding_row=1 if is_first_block else 0,
            read_mode=0,
            transnum=15,
            line_buffer_idx=0,
            src_buffer_idx='a',
            bas_addr=base_addr_0,
        )
        DataLoader.dispatch(
            layer_idx=hw_layer_idx,
            line_buffer_reshape=0,
            is_padding_row=0,
            read_mode=0,
            transnum=15,
            line_buffer_idx=1,
            src_buffer_idx='b',
            bas_addr=base_addr_1,
        )
        WeightLoader.dispatch(
            acc_reg_comp_idx=0,
            kernal_size=0,
            line_buffer_row_shift=1,
            line_buffer_idx=0,
            is_padding_col=1,
            weight_parall_mode=0,
            is_new=1 if is_first_block else 0,
            transnum=9,
            bas_addr=0,
            is_bilinear_bicubic=1,
            offset_reg_idx=0,
        )
        WeightLoader.dispatch(
            acc_reg_comp_idx=1,
            kernal_size=0,
            line_buffer_row_shift=1,
            line_buffer_idx=1,
            is_padding_col=1,
            weight_parall_mode=0,
            is_new=1 if is_first_block else 0,
            transnum=9,
            bas_addr=0,
            is_bilinear_bicubic=1,
            offset_reg_idx=0,
        )
        DataStorer.dispatch(
            quant_config_idx=0,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=0,
            acc_mode=0,
            transfer_num=1,
            store_mode=0,
            stride=STRIDE,
            base_addr_pooling=0,
            base_addrs_res=base_addrs_res_0,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx='b',
        )
        DataStorer.dispatch(
            quant_config_idx=0,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=1,
            acc_mode=0,
            transfer_num=1,
            store_mode=0,
            stride=STRIDE,
            base_addr_pooling=0,
            base_addrs_res=base_addrs_res_1,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx='b',
        )


def generate_usrnet_instructions_unrolled(layer_info: list, params: dict) -> list:
    """Generate UNROLLED instructions for USRNet (conv/deformable unrolled per row pair)."""
    from instruction import OffchipDataLoader

    logger.info("Generating USRNet UNROLLED instructions...")

    OffchipDataLoader.dispatch(
        transnum=100,
        load_model=0,
        src_buffer_idx=2,
        bas_addr=0,
    )

    for idx, layer in enumerate(layer_info):
        layer_type = layer['type']

        if 'conv2d' in layer_type:
            _generate_conv_instructions_unrolled(idx, layer)
        elif 'deformable' in layer_type:
            _generate_deformable_conv_instructions_unrolled(idx, layer)
        elif layer_type in ['nn.relu', 'nn.leaky_relu', 'sigmoid', 'tanh']:
            _generate_activation_instructions(idx, layer)
        elif layer_type in ['add', 'subtract', 'multiply']:
            _generate_elementwise_instructions(idx, layer)
        elif layer_type in ['nn.batch_norm', 'nn.layer_norm']:
            _generate_norm_instructions(idx, layer)
        elif layer_type in ['nn.max_pool2d', 'nn.avg_pool2d']:
            _generate_pooling_instructions(idx, layer)
        elif layer_type == 'concatenate':
            _generate_concat_instructions(idx, layer)
        elif layer_type in ['nn.upsampling', 'image.resize2d']:
            _generate_upsample_instructions(idx, layer)
        elif layer_type == 'clip':
            _generate_clip_instructions(idx, layer)
        elif layer_type in ['reshape', 'transpose', 'squeeze', 'expand_dims']:
            _generate_layout_transform_instructions(idx, layer)

    return Inst.code_list


def compile_usrnet_manual_unrolled(model_path: str, output_dir: str):
    """Compile USRNet with unrolled instructions; output to output_dir (usr_net_unroll)."""
    logger.info("=" * 60)
    logger.info("USRNet UNROLLED Compilation")
    logger.info("=" * 60)

    logger.info("\n[1/5] Importing ONNX model...")
    importer = ONNXImporter()
    mod, params = importer.import_model(model_path)
    logger.info(f"✓ Imported model with {len(params)} parameters")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/01_imported_manual.txt", "w") as f:
        f.write(str(mod))

    logger.info("\n[2/5] Applying optimization passes...")
    mod = PassManager.optimize(
        mod,
        preset='vis_vpu',
        opt_level=3,
        enable_fold_constant=False,
        enable_fold_scale_axis=False,
    )
    with open(f"{output_dir}/02_optimized_manual.txt", "w") as f:
        f.write(str(mod))

    logger.info("\n[3/5] Analyzing USRNet structure...")
    layer_info = analyze_usrnet_structure(mod, params)
    logger.info(f"✓ Found {len(layer_info)} layers")

    logger.info("\n[4/5] Generating UNROLLED instructions...")
    Inst.code_list = []
    Inst.current_code_num = 0
    instructions = generate_usrnet_instructions_unrolled(layer_info, params)
    logger.info(f"✓ Generated {len(instructions)} instructions (unrolled)")

    logger.info("\n[5/5] Packaging...")
    inst_file = f"{output_dir}/usr_net_unroll_inst.txt"
    with open(inst_file, "w") as f:
        for inst in instructions:
            f.write(str(inst) + "\n")
    logger.info(f"✓ Wrote {inst_file}")

    binary_file = f"{output_dir}/usr_net_unroll_inst.bin"
    try:
        from assembler import compile_file
        compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
        logger.info(f"✓ Assembled {binary_file}")
    except Exception as e:
        logger.error(f"Assembly failed: {e}")

    param_file = f"{output_dir}/usr_net_unroll_params.json"
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
    with open(param_file, "w") as f:
        json.dump(param_info, f, indent=2)
    logger.info(f"✓ Saved {param_file}")

    return {
        "instructions": inst_file,
        "binary": binary_file,
        "params": param_file,
    }


if __name__ == "__main__":
    MODEL_PATH = "../USR_Net.onnx"
    OUTPUT_DIR = "./output/usr_net_unroll"

    print("\n" + "=" * 70)
    print("USRNet UNROLLED Compilation (output: usr_net_unroll)")
    print("=" * 70 + "\n")

    try:
        result = compile_usrnet_manual_unrolled(MODEL_PATH, OUTPUT_DIR)
        print("\n✓ Unrolled compilation completed successfully!")
        print("\nGenerated outputs:")
        for key, path in result.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  {key:12s}: {path} ({size} bytes)")
        print("\nNext steps:")
        print("  1. Compare with golden: python compare_with_golden.py --mine output/usr_net_unroll/usr_net_unroll_inst.txt")
        print("  2. Estimate cycles: python estimate_cycles.py output/usr_net_unroll/usr_net_unroll_inst.txt")
    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=True)
        sys.exit(1)
