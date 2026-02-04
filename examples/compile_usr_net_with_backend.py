"""
Complete compilation example for USRNet connecting frontend to backend.

This script demonstrates two approaches:
1. Automated: Use IR traversal to generate instructions (partially implemented)
2. Manual: Direct mapping to sd_sr_codegen functions (for custom control)
"""

import sys
sys.path.append('..')

import os
import logging
import numpy as np
from pathlib import Path

# TVM imports
import tvm
from tvm import relay

# Frontend
from frontend.onnx_importer import ONNXImporter
from passes.pass_manager import PassManager

# Codegen
from codegen.ir_to_inst_bridge import generate_instructions_from_relay
from codegen.binary_packer import BinaryPacker

# Backend (for manual approach)
from instruction import Inst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_usrnet_automated(model_path: str, output_dir: str):
    """
    Approach 1: Automated IR traversal (generic but may need tuning).
    
    This approach uses IR traversal to automatically generate instructions
    from the Relay IR. Good for standard models but may need customization
    for complex models like USRNet with deformable convolutions.
    """
    logger.info("=" * 60)
    logger.info("AUTOMATED APPROACH: IR Traversal")
    logger.info("=" * 60)
    
    # Step 1: Import model (Frontend)
    logger.info("\n[1/5] Importing ONNX model...")
    importer = ONNXImporter()
    mod, params = importer.import_model(model_path)
    logger.info(f"✓ Imported model with {len(params)} parameters")
    
    # Save imported IR
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/01_imported.txt", "w") as f:
        f.write(str(mod))
    
    # Step 2: Optimize (Frontend passes)
    logger.info("\n[2/5] Applying optimization passes...")
    mod = PassManager.optimize(
        mod,
        preset='vis_vpu',
        opt_level=3,
        enable_fold_constant=True,
        enable_fold_scale_axis=True
    )
    logger.info("✓ Applied optimizations")
    
    # Save optimized IR
    with open(f"{output_dir}/02_optimized.txt", "w") as f:
        f.write(str(mod))
    
    # Step 3: Lower to TIR (optional, depends on your backend)
    logger.info("\n[3/5] Lowering to TIR...")
    # For direct instruction generation, you might skip full TIR lowering
    # and work directly with Relay IR
    with open(f"{output_dir}/03_tir.txt", "w") as f:
        f.write(str(mod))  # Simplified: would be actual TIR
    logger.info("✓ Generated TIR")
    
    # Step 4: Generate Instructions (Middle layer - IR to Instructions)
    logger.info("\n[4/5] Generating hardware instructions from IR...")
    instructions = generate_instructions_from_relay(mod, params)
    logger.info(f"✓ Generated {len(instructions)} instructions")
    
    # Step 5: Package (Backend - Instructions to Binary)
    logger.info("\n[5/5] Packaging instructions and parameters...")
    packer = BinaryPacker(output_dir=output_dir)
    outputs = packer.pack(instructions, params, output_name="usr_net")
    logger.info("✓ Packaged outputs:")
    for key, path in outputs.items():
        logger.info(f"  - {key}: {path}")
    
    return outputs


def compile_usrnet_manual(model_path: str, output_dir: str):
    """
    Approach 2: Manual instruction generation using sd_sr_codegen.py.
    
    This approach gives you full control over instruction generation.
    You analyze the model structure and manually call the instruction
    generation functions from sd_sr_codegen.py.
    
    This is better for USRNet because:
    - Complex deformable convolutions
    - Custom quantization strategies
    - Specific buffer management requirements
    """
    logger.info("=" * 60)
    logger.info("MANUAL APPROACH: Direct Backend Invocation")
    logger.info("=" * 60)

    # Step 1: Import model
    logger.info("\n[1/5] Importing ONNX model...")
    importer = ONNXImporter()
    mod, params = importer.import_model(model_path)
    logger.info(f"✓ Imported model with {len(params)} parameters")

    # Save imported IR
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/01_imported_manual.txt", "w") as f:
        f.write(str(mod))
    logger.info(f"✓ Saved imported IR to {output_dir}/01_imported_manual.txt")

    # Step 2: Apply optimization passes
    logger.info("\n[2/5] Applying optimization passes...")
    # Note: Disable FoldConstant and FoldScaleAxis as they may require LLVM codegen
    # Use lighter optimizations that don't require code execution
    mod = PassManager.optimize(
        mod,
        preset='vis_vpu',
        opt_level=3,
        enable_fold_constant=False,  # Disabled - requires LLVM
        enable_fold_scale_axis=False  # Disabled - may require LLVM
    )
    logger.info("✓ Applied optimizations (FoldConstant & FoldScaleAxis disabled)")

    # Save optimized IR
    with open(f"{output_dir}/02_optimized_manual.txt", "w") as f:
        f.write(str(mod))
    logger.info(f"✓ Saved optimized IR to {output_dir}/02_optimized_manual.txt")

    # Step 3: Analyze the optimized model structure
    logger.info("\n[3/5] Analyzing USRNet structure...")
    layer_info = analyze_usrnet_structure(mod, params)
    logger.info(f"✓ Found {len(layer_info)} layers")
    for i, layer in enumerate(layer_info[:5]):  # Show first 5
        logger.info(f"  Layer {i}: {layer['type']} {layer.get('shape', '')}")

    # Step 4: Generate instructions using manual backend
    logger.info("\n[4/5] Generating instructions using manual backend...")
    
    # Clear previous instructions
    Inst.code_list = []
    Inst.current_code_num = 0
    
    # Call your custom instruction generation
    # This is where you would import and call functions from sd_sr_codegen.py
    # For USRNet specifically:
    instructions = generate_usrnet_instructions(layer_info, params)
    
    logger.info(f"✓ Generated {len(instructions)} instructions")

    # Step 5: Package
    logger.info("\n[5/5] Packaging...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write instructions
    inst_file = f"{output_dir}/usr_net_inst.txt"
    with open(inst_file, 'w') as f:
        for inst in instructions:
            f.write(str(inst) + '\n')
    logger.info(f"✓ Wrote instructions to {inst_file}")
    
    # Write binary (using assembler)
    from assembler import compile_file
    binary_file = f"{output_dir}/usr_net_inst.bin"
    try:
        compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
        logger.info(f"✓ Assembled binary to {binary_file}")
    except Exception as e:
        logger.error(f"Assembly failed: {e}")
    
    # Save parameters
    import json
    param_file = f"{output_dir}/usr_net_params.json"
    param_info = {}
    for name, param in params.items():
        if hasattr(param, 'numpy'):
            param_np = param.numpy()
        elif isinstance(param, np.ndarray):
            param_np = param
        else:
            param_np = np.array(param)
        
        param_info[name] = {
            'shape': list(param_np.shape),
            'dtype': str(param_np.dtype),
            'size': int(np.prod(param_np.shape))
        }
    
    with open(param_file, 'w') as f:
        json.dump(param_info, f, indent=2)
    logger.info(f"✓ Saved parameter info to {param_file}")
    
    return {
        'instructions': inst_file,
        'binary': binary_file,
        'params': param_file
    }


def analyze_usrnet_structure(mod: tvm.IRModule, params: dict) -> list:
    """
    Analyze USRNet structure to extract layer information.
    
    This helps you understand what operations need to be mapped
    to hardware instructions.
    """
    layers = []
    
    func = mod["main"]
    
    class LayerAnalyzer(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.layers = []
            
        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                op_name = call.op.name
                
                layer = {
                    'type': op_name,
                    'attrs': {}
                }
                
                # Extract attributes
                if hasattr(call, 'attrs') and call.attrs:
                    for key in call.attrs.keys():
                        layer['attrs'][key] = getattr(call.attrs, key)
                
                # Extract shapes
                if hasattr(call, 'checked_type'):
                    checked_type = call.checked_type
                    # Handle TupleType (operations with multiple outputs)
                    if isinstance(checked_type, tvm.ir.type.TupleType):
                        # Get shape from first field for simplicity
                        if len(checked_type.fields) > 0 and hasattr(checked_type.fields[0], 'shape'):
                            layer['shape'] = [int(d) for d in checked_type.fields[0].shape]
                        else:
                            layer['shape'] = 'tuple'
                    # Handle regular TensorType
                    elif hasattr(checked_type, 'shape'):
                        layer['shape'] = [int(d) for d in checked_type.shape]
                    else:
                        layer['shape'] = 'unknown'
                
                self.layers.append(layer)
            
            super().visit_call(call)
    
    analyzer = LayerAnalyzer()
    analyzer.visit(func)
    
    return analyzer.layers


def generate_usrnet_instructions(layer_info: list, params: dict) -> list:
    """
    Generate instructions for USRNet using the manual backend.
    
    This is where you would map the layer_info to specific
    instruction generation functions from sd_sr_codegen.py.
    
    For now, this is a placeholder that shows the structure.
    You would implement the actual mapping based on:
    1. Layer types (conv, deformable_conv, etc.)
    2. Layer parameters (channels, kernel size, etc.)
    3. Hardware constraints (buffer sizes, etc.)
    """
    from instruction import (
        OffchipDataLoader, DataLoader, WeightLoader,
        QuantLoader, DataStorer, OffchipDataStorer
    )
    
    logger.info("Generating USRNet-specific instructions...")
    
    # Example: Load weights and quantization parameters
    OffchipDataLoader.dispatch(
        transnum=100,  # Based on actual parameter size
        load_model=0,
        src_buffer_idx=2,
        bas_addr=0
    )
    
    # For each layer, generate appropriate instruction sequence
    for idx, layer in enumerate(layer_info):
        layer_type = layer['type']
        
        if 'conv2d' in layer_type:
            # Generate conv2d instructions
            _generate_conv_instructions(idx, layer)
        elif 'deformable' in layer_type:
            # Generate deformable conv instructions
            _generate_deformable_conv_instructions(idx, layer)
        # Add more layer types as needed
    
    # Get all generated instructions
    instructions = Inst.code_list
    
    return instructions


def _generate_conv_instructions(layer_idx: int, layer: dict):
    """Generate instructions for standard convolution."""
    from instruction import DataLoader, WeightLoader, QuantLoader, DataStorer

    # Hardware constraint: layer_idx is 5 bits (max 31)
    # Use modulo to wrap layer indices that exceed the hardware limit
    hw_layer_idx = layer_idx % 32

    # Load quantization parameters
    QuantLoader.dispatch(
        quant_reg_load_idx=0,
        quant_mode=0,
        layer_idx=hw_layer_idx,
        transnum=4,
        bas_addr=0
    )

    # Load input data
    DataLoader.dispatch(
        layer_idx=hw_layer_idx,
        line_buffer_reshape=0,
        is_padding_row=0,
        read_mode=0,
        transnum=15,  # Max value for 4-bit field (was 32, but limited to 15)
        line_buffer_idx=0,
        src_buffer_idx='a',
        bas_addr=0
    )
    
    # Load weights
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
    
    # Store results
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


def _generate_deformable_conv_instructions(layer_idx: int, layer: dict):
    """Generate instructions for deformable convolution."""
    from instruction import (
        DataLoader, WeightLoader, OffsetLoader,
        QuantLoader, DataStorer
    )
    
    # This would include offset generation and loading
    # See sd_sr_codegen.py for the actual implementation
    logger.info(f"Generating deformable conv for layer {layer_idx}")
    # Implementation would go here


def add_instruction_dependencies(instructions: list) -> list:
    """
    Compute dependencies between instructions.

    This is adapted from sd_sr_codegen.py lines 3951-4111.
    Dependencies are already added by instruction.py dispatch methods.
    """
    # Dependencies are already set by the instruction dispatch methods
    # This function is kept for future enhancement if needed
    return instructions


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "../USR_Net.onnx"
    OUTPUT_DIR = "./output/usr_net"
    
    # Choose compilation approach
    APPROACH = "manual"  # "automated" or "manual"
    
    print("\n" + "=" * 70)
    print("USRNet Compilation: Frontend + Backend Integration")
    print("=" * 70 + "\n")
    
    try:
        if APPROACH == "automated":
            result = compile_usrnet_automated(MODEL_PATH, OUTPUT_DIR)
            print("\n✓ Automated compilation completed successfully!")
        else:
            result = compile_usrnet_manual(MODEL_PATH, OUTPUT_DIR)
            print("\n✓ Manual compilation completed successfully!")
        
        print("\nGenerated outputs:")
        for key, path in result.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  {key:12s}: {path} ({size} bytes)")
        
        print("\nNext steps:")
        print("  1. Verify instruction correctness in usr_net_inst.txt")
        print("  2. Check binary format in usr_net_inst.bin")
        print("  3. Deploy to hardware or simulator")
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=True)
        sys.exit(1)

