"""
Compile FSRCNN to hardware instructions (bridge frontend -> backend).

Same model loading as compile_fsrcnn.py: instantiate PyTorch FSRCNN and
example input, then convert to Relay via ModelLoader (no ONNX export/import).
- Optimize with PassManager
- Analyze layer structure and generate instructions via instruction.*
- Emit fsrcnn_inst.txt, fsrcnn_inst.bin, fsrcnn_params.json

FSRCNN: input -> conv -> prelu -> [conv -> prelu]* -> conv -> pixel_shuffle.
"""

import sys
import os
sys.path.append('..')

import json
import logging
import numpy as np
import torch

import tvm
from tvm import relay

from models_new_930 import FSRCNN
from frontend.model_loader import ModelLoader
from passes.pass_manager import PassManager
from instruction import (
    Inst, OffchipDataLoader, DataLoader, WeightLoader,
    QuantLoader, DataStorer, OffsetLoader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fsrcnn(weights_path: str = None, example_input_shape=(1, 1, 270, 480)):
    """
    Load FSRCNN as (mod, params) the same way as compile_fsrcnn.py:
    instantiate FSRCNN, optional load weights from file, create example input,
    then ModelLoader.load(model, model_format='pytorch', example_inputs=...).
    """
    logger.info("Loading FSRCNN (PyTorch model + example input)...")
    model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
    model.eval()

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", weights_path)

    example_input = torch.randn(*example_input_shape)
    loader = ModelLoader()
    mod, params = loader.load(
        model,
        model_format='pytorch',
        example_inputs=example_input
    )
    return mod, params


def _tensor_type_shape_to_list(shape):
    """Relay TensorType.shape entries may be int, IntImm, tir.Any, etc."""
    if shape is None:
        return None
    out = []
    for d in shape:
        if isinstance(d, tvm.tir.Any):
            out.append(None)
        elif isinstance(d, int):
            out.append(d)
        else:
            try:
                out.append(int(d))
            except (TypeError, ValueError):
                out.append(None)
    return out


def analyze_fsrcnn_structure(mod: tvm.IRModule, params: dict) -> list:
    """Extract layer list from Relay main (same pattern as analyze_usrnet_structure)."""
    func = mod["main"]

    class LayerAnalyzer(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.layers = []

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                op_name = call.op.name
                layer = {'type': op_name, 'attrs': {}}
                if hasattr(call, 'attrs') and call.attrs:
                    for key in call.attrs.keys():
                        layer['attrs'][key] = getattr(call.attrs, key)
                if hasattr(call, 'checked_type'):
                    ct = call.checked_type
                    if isinstance(ct, tvm.ir.type.TupleType) and len(ct.fields) > 0:
                        f = ct.fields[0]
                        layer['shape'] = (
                            _tensor_type_shape_to_list(f.shape)
                            if hasattr(f, 'shape') else 'tuple'
                        )
                    elif hasattr(ct, 'shape'):
                        layer['shape'] = _tensor_type_shape_to_list(ct.shape)
                    else:
                        layer['shape'] = 'unknown'
                self.layers.append(layer)
            super().visit_call(call)

    analyzer = LayerAnalyzer()
    analyzer.visit(func)
    return analyzer.layers


# Shape/index/metadata ops: no VPU micro-op sequence (CPU handles or folded away).
_FSRCNN_SKIP_OPS = frozenset({
    'broadcast_to',
    'shape_of',
    'cast',
    'cast_like',
    'less',
    'greater_equal',
    'where',
    'scatter',
    'slice_like',
})


def generate_fsrcnn_instructions(layer_info: list, params: dict) -> list:
    """
    Generate hardware instructions for FSRCNN.
    - OffchipDataLoader once (input/weights).
    - For each conv: QuantLoader -> DataLoader -> WeightLoader -> DataStorer.
    - Deformable conv adds OffsetLoader before weights.
    - Activations, elementwise, pooling, clip, layout ops mirror USRNet backend style.
    - Last nn.conv2d (not deformable) may use pixel shuffle on DataStorer.
    - Shape/slice index subgraph ops emit no instructions.
    """
    logger.info("Generating FSRCNN instructions (%d layers)...", len(layer_info))

    # Initial offchip load (input / weights)
    OffchipDataLoader.dispatch(
        transnum=100,
        load_model=0,
        src_buffer_idx=2,
        bas_addr=0
    )

    # Last standard conv2d only (deformable name also contains "conv2d")
    conv_indices = [
        i for i, L in enumerate(layer_info)
        if 'conv2d' in L['type'] and 'deformable' not in L['type']
    ]
    last_conv_idx = conv_indices[-1] if conv_indices else -1

    for idx, layer in enumerate(layer_info):
        layer_type = layer['type']

        if layer_type in _FSRCNN_SKIP_OPS:
            continue
        if layer_type == 'dyn.strided_slice':
            continue
        if 'deformable' in layer_type:
            _generate_fsrcnn_deformable_conv_instructions(idx, layer)
        elif 'conv2d' in layer_type:
            use_pixel_shuffle = idx == last_conv_idx
            _generate_fsrcnn_conv_instructions(idx, layer, use_pixel_shuffle)
        elif layer_type in (
            'nn.relu', 'nn.leaky_relu', 'sigmoid', 'tanh', 'nn.prelu',
        ):
            _generate_fsrcnn_activation_instructions(idx, layer)
        elif layer_type in ('add', 'subtract', 'multiply'):
            _generate_fsrcnn_elementwise_instructions(idx, layer)
        elif layer_type == 'nn.bias_add':
            _generate_fsrcnn_bias_add_instructions(idx, layer)
        elif layer_type in ('nn.batch_norm', 'nn.layer_norm'):
            _generate_fsrcnn_norm_instructions(idx, layer)
        elif layer_type in ('nn.max_pool2d', 'nn.avg_pool2d'):
            _generate_fsrcnn_pooling_instructions(idx, layer)
        elif layer_type == 'concatenate':
            _generate_fsrcnn_concat_instructions(idx, layer)
        elif layer_type in ('nn.upsampling', 'image.resize2d'):
            _generate_fsrcnn_upsample_instructions(idx, layer)
        elif layer_type == 'clip':
            _generate_fsrcnn_clip_instructions(idx, layer)
        elif layer_type in (
            'reshape', 'transpose', 'squeeze', 'expand_dims', 'repeat',
        ):
            _generate_fsrcnn_layout_instructions(idx, layer)
        elif 'pad' in layer_type:
            _generate_fsrcnn_layout_instructions(idx, layer)
        else:
            logger.warning(
                "FSRCNN: no dedicated handler for op %r at layer_idx %s; "
                "emitting pass-through DataLoader -> DataStorer",
                layer_type,
                idx,
            )
            _generate_fsrcnn_activation_instructions(idx, layer)

    instructions = Inst.code_list
    return instructions


def _generate_fsrcnn_conv_instructions(layer_idx: int, layer: dict, use_pixel_shuffle: bool):
    """Single conv: QuantLoader -> DataLoader -> WeightLoader -> DataStorer (optional pixel shuffle)."""
    hw_layer_idx = layer_idx % 32

    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0
    )
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0
    )
    WeightLoader.dispatch(
        acc_reg_comp_idx=0,
        kernal_size=0,
        line_buffer_row_shift=1,
        line_buffer_idx=0,
        is_padding_col=1,
        weight_parall_mode=0,
        is_new=0,
        transnum=9,
        bas_addr=0,
        is_bilinear_bicubic=0,
        offset_reg_idx=0
    )
    DataStorer.dispatch(
        quant_config_idx=0,
        pixelshuffle_out_mode=1 if use_pixel_shuffle else 0,
        is_pixelshuffle=1 if use_pixel_shuffle else 0,
        pooling_out_mode=0,
        pooling_out_new=0,
        is_pooling=0,
        reg_out_idx=0,
        acc_mode=0,
        transfer_num=1,
        store_mode=0,
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx=(
            'fsrcnn_output_buffer' if use_pixel_shuffle else 'b'
        )
    )


def _generate_fsrcnn_activation_instructions(layer_idx: int, layer: dict):
    """Activations usually fused; if standalone: DataLoader -> DataStorer (pass-through)."""
    hw_layer_idx = layer_idx % 32
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0
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
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b'
    )


def _generate_fsrcnn_layout_instructions(layer_idx: int, layer: dict):
    """Pad / reshape / transpose / squeeze / expand_dims / repeat: loader + storer with reshape hints."""
    hw_layer_idx = layer_idx % 32
    layer_type = layer['type']
    reshape_mode = 0
    if 'transpose' in layer_type:
        reshape_mode = 1
    elif layer_type in ('reshape', 'repeat') or 'pad' in layer_type:
        reshape_mode = 2
    elif layer_type in ('squeeze', 'expand_dims'):
        reshape_mode = 2

    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=reshape_mode,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0
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
        store_mode=1 if reshape_mode > 0 else 0,
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b'
    )


def _generate_fsrcnn_deformable_conv_instructions(layer_idx: int, layer: dict):
    """Deformable conv: QuantLoader -> OffsetLoader -> DataLoader -> WeightLoader -> DataStorer."""
    hw_layer_idx = layer_idx % 32

    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0,
    )
    OffsetLoader.dispatch(offset_reg_idx=0, bas_addr=0)
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0,
    )
    WeightLoader.dispatch(
        acc_reg_comp_idx=0,
        kernal_size=0,
        line_buffer_row_shift=1,
        line_buffer_idx=0,
        is_padding_col=1,
        weight_parall_mode=0,
        is_new=0,
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
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b',
    )


def _generate_fsrcnn_elementwise_instructions(layer_idx: int, layer: dict):
    """Element-wise: two DataLoaders -> DataStorer (acc_mode add/multiply)."""
    hw_layer_idx = layer_idx % 32
    layer_type = layer['type']
    if layer_type == 'add':
        acc_mode = 1
    elif layer_type == 'multiply':
        acc_mode = 3
    else:
        acc_mode = 0

    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0,
    )
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=1,
        src_buffer_idx='b',
        bas_addr=0,
    )
    DataStorer.dispatch(
        quant_config_idx=0,
        pixelshuffle_out_mode=0,
        is_pixelshuffle=0,
        pooling_out_mode=0,
        pooling_out_new=0,
        is_pooling=0,
        reg_out_idx=0,
        acc_mode=acc_mode,
        transfer_num=1,
        store_mode=0,
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='a',
    )


def _generate_fsrcnn_bias_add_instructions(layer_idx: int, layer: dict):
    """Bias add on feature map: pass-through load/store (bias fused with conv when possible)."""
    _generate_fsrcnn_activation_instructions(layer_idx, layer)


def _generate_fsrcnn_norm_instructions(layer_idx: int, layer: dict):
    """Norm params via QuantLoader (typically fused with conv at deploy time)."""
    hw_layer_idx = layer_idx % 32
    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0,
    )


def _generate_fsrcnn_pooling_instructions(layer_idx: int, layer: dict):
    """Pooling: DataLoader -> DataStorer with is_pooling."""
    hw_layer_idx = layer_idx % 32
    attrs = layer.get('attrs', {})
    strides = attrs.get('strides', [1, 1])
    pool_mode = 1 if 'avg' in layer['type'] else 0

    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0,
    )
    DataStorer.dispatch(
        quant_config_idx=0,
        pixelshuffle_out_mode=0,
        is_pixelshuffle=0,
        pooling_out_mode=pool_mode,
        pooling_out_new=0,
        is_pooling=1,
        reg_out_idx=0,
        acc_mode=0,
        transfer_num=1,
        store_mode=0,
        stride=strides[1] if len(strides) > 1 else 1,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b',
    )


def _generate_fsrcnn_concat_instructions(layer_idx: int, layer: dict):
    """Concat is buffer layout; no extra opcode (same as USRNet manual path)."""
    logger.debug("FSRCNN layer %s: concatenate (buffer layout only)", layer_idx)


def _generate_fsrcnn_upsample_instructions(layer_idx: int, layer: dict):
    """Resize / upsample: optional pixel shuffle or bicubic flags on storer."""
    hw_layer_idx = layer_idx % 32
    attrs = layer.get('attrs', {})
    method = attrs.get('method', 'nearest_neighbor')
    use_pixel_shuffle = (
        'depth_to_space' in str(attrs) or method == 'pixel_shuffle'
    )
    use_bilinear = 'bilinear' in str(method) or 'bicubic' in str(method)

    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0,
    )
    DataStorer.dispatch(
        quant_config_idx=0,
        pixelshuffle_out_mode=0 if not use_pixel_shuffle else 1,
        is_pixelshuffle=1 if use_pixel_shuffle else 0,
        pooling_out_mode=0,
        pooling_out_new=0,
        is_pooling=0,
        reg_out_idx=0,
        acc_mode=0,
        transfer_num=1,
        store_mode=0,
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=1 if use_bilinear else 0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b',
    )


def _generate_fsrcnn_clip_instructions(layer_idx: int, layer: dict):
    """Clip / clamp: load + store (limits via quant path in hardware)."""
    hw_layer_idx = layer_idx % 32
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0,
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
        stride=32,
        base_addr_pooling=0,
        base_addrs_res=0,
        is_bicubic_add=0,
        is_first_or_last_row=0,
        is_mask=0,
        is_new=0,
        dest_buffer_idx='b',
    )


def compile_fsrcnn_with_backend(output_dir: str, weights_path: str = None):
    """
    Full pipeline: load FSRCNN (same as compile_fsrcnn.py) -> optimize ->
    analyze -> generate instructions -> package.
    """
    logger.info("=" * 60)
    logger.info("FSRCNN: Frontend + Backend (hardware instructions)")
    logger.info("=" * 60)

    # 1) Load (instantiate model + example input, no ONNX)
    logger.info("\n[1/5] Loading model...")
    mod, params = load_fsrcnn(weights_path=weights_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "01_imported.txt"), "w") as f:
        f.write(str(mod))
    logger.info("Saved 01_imported.txt")

    # 2) Optimize
    logger.info("\n[2/5] Optimizing...")
    mod = PassManager.optimize(
        mod,
        preset='vis_vpu',
        opt_level=3,
        enable_fold_constant=False,
        enable_fold_scale_axis=False
    )
    with open(os.path.join(output_dir, "02_optimized.txt"), "w") as f:
        f.write(str(mod))
    logger.info("Saved 02_optimized.txt")

    # 3) Analyze
    logger.info("\n[3/5] Analyzing FSRCNN structure...")
    layer_info = analyze_fsrcnn_structure(mod, params)
    logger.info("Layers: %d", len(layer_info))
    for i, L in enumerate(layer_info[:8]):
        logger.info("  %d: %s %s", i, L['type'], L.get('shape', ''))

    # 4) Generate instructions
    logger.info("\n[4/5] Generating instructions...")
    Inst.code_list = []
    Inst.current_code_num = 0
    instructions = generate_fsrcnn_instructions(layer_info, params)
    logger.info("Generated %d instructions", len(instructions))

    # 5) Package
    logger.info("\n[5/5] Packaging...")
    inst_file = os.path.join(output_dir, "fsrcnn_inst.txt")
    with open(inst_file, 'w') as f:
        for inst in instructions:
            f.write(str(inst) + '\n')
    logger.info("Wrote %s", inst_file)

    binary_file = os.path.join(output_dir, "fsrcnn_inst.bin")
    try:
        from assembler import compile_file
        compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
        logger.info("Assembled %s", binary_file)
    except Exception as e:
        logger.error("Assembly failed: %s", e)

    param_file = os.path.join(output_dir, "fsrcnn_params.json")
    param_info = {}
    for name, param in params.items():
        if hasattr(param, 'numpy'):
            arr = param.numpy()
        elif isinstance(param, np.ndarray):
            arr = param
        else:
            arr = np.array(param)
        param_info[name] = {
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'size': int(np.prod(arr.shape))
        }
    with open(param_file, 'w') as f:
        json.dump(param_info, f, indent=2)
    logger.info("Saved %s", param_file)

    return {
        'instructions': inst_file,
        'binary': binary_file,
        'params': param_file
    }


if __name__ == '__main__':
    # Same pattern as compile_fsrcnn.py: no model file required.
    # Optional: set FSRCNN_WEIGHTS to load weights from .pth
    _dir = os.path.dirname(__file__)
    OUTPUT_DIR = os.path.join(_dir, "output", "fsrcnn")
    WEIGHTS_PATH = os.environ.get("FSRCNN_WEIGHTS")  # optional .pth

    print("\n" + "=" * 70)
    print("FSRCNN Compilation: Frontend + Backend (hardware instructions)")
    print("=" * 70 + "\n")

    try:
        result = compile_fsrcnn_with_backend(OUTPUT_DIR, weights_path=WEIGHTS_PATH)
        print("\n✓ Compilation completed successfully!")
        print("\nGenerated outputs:")
        for key, path in result.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print("  %s: %s (%s bytes)" % (key, path, size))
        print("\nNext: run simulator on", result.get('instructions'))
    except Exception:
        logger.exception("Compilation failed")
        sys.exit(1)
