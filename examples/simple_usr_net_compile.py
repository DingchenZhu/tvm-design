"""
Simple USRNet compilation script - connects frontend IR to backend instructions.

This is a minimal example showing how to bridge the gap between your
TVM frontend (which you've already run) and the instruction backend.
"""

import sys
sys.path.append('..')

import os
import json
import logging
from pathlib import Path

# Import instruction API (backend)
from instruction import Inst

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_existing_ir(output_dir: str):
    """
    Analyze the existing IR files you've already generated.
    
    You already have:
    - 01_imported.txt
    - 02_optimized.txt  
    - 03_tir.txt
    
    This function reads them to understand model structure.
    """
    logger.info(f"Analyzing IR in {output_dir}")
    
    # Read the optimized IR
    ir_file = f"{output_dir}/02_optimized.txt"
    if not os.path.exists(ir_file):
        logger.error(f"IR file not found: {ir_file}")
        logger.info("Run the frontend compilation first!")
        return None
    
    with open(ir_file, 'r') as f:
        ir_content = f.read()
    
    logger.info(f"Loaded IR ({len(ir_content)} bytes)")
    
    # Simple parsing to extract layer information
    layers = []
    
    # Count convolutions
    conv_count = ir_content.count('nn.conv2d')
    dense_count = ir_content.count('nn.dense')
    pool_count = ir_content.count('pool2d')
    
    logger.info(f"Found operators: {conv_count} convs, {dense_count} dense, {pool_count} pools")
    
    return {
        'conv_count': conv_count,
        'dense_count': dense_count,
        'pool_count': pool_count,
        'ir_content': ir_content
    }


def generate_instructions_for_usrnet(model_info: dict):
    """
    Generate instructions based on model structure.
    
    For USRNet, you would adapt the pattern from sd_sr_codegen.py.
    This is a simplified example showing the structure.
    """
    from instruction import (
        OffchipDataLoader, DataLoader, WeightLoader,
        QuantLoader, DataStorer, OffchipDataStorer
    )
    
    logger.info("Generating instructions...")
    
    # Clear previous instructions
    Inst.code_list = []
    Inst.current_code_num = 0
    
    # ========================================
    # Step 1: Load off-chip data
    # ========================================
    logger.info("  [1/4] Off-chip data loading...")
    
    # Load quantization parameters
    OffchipDataLoader.dispatch(
        transnum=100,  # Adjust based on actual size
        load_model=0,
        src_buffer_idx=2,  # Quant buffer
        bas_addr=0
    )
    
    # Load weights
    OffchipDataLoader.dispatch(
        transnum=500,  # Adjust based on actual size
        load_model=0,
        src_buffer_idx=1,  # Weight buffer
        bas_addr=0
    )
    
    # Load input image
    OffchipDataLoader.dispatch(
        transnum=144*4,  # Example: 256x144 image
        load_model=0,
        src_buffer_idx=0,  # Input buffer
        bas_addr=0
    )
    
    # ========================================
    # Step 2: Generate layer-by-layer instructions
    # ========================================
    logger.info("  [2/4] Layer instructions...")
    
    num_convs = model_info.get('conv_count', 10)
    
    for layer_idx in range(min(num_convs, 3)):  # Start with first 3 layers
        logger.info(f"    Layer {layer_idx}: Conv2D")
        
        # Load quantization config for this layer
        QuantLoader.dispatch(
            quant_reg_load_idx=layer_idx % 2,  # Ping-pong between 0 and 1
            quant_mode=0,  # Standard mode
            layer_idx=layer_idx,
            transnum=4,
            bas_addr=layer_idx * 4
        )
        
        # Example: process 32 rows
        for row_idx in range(32):
            # Load data
            DataLoader.dispatch(
                layer_idx=layer_idx,
                line_buffer_reshape=0,
                is_padding_row=1 if row_idx == 0 else 0,  # Padding at top
                read_mode=0,
                transnum=32,  # Load 32 elements
                line_buffer_idx=row_idx % 2,  # Ping-pong
                src_buffer_idx='a' if layer_idx % 2 == 0 else 'b',
                bas_addr=row_idx * 32
            )
            
            # Load weights
            WeightLoader.dispatch(
                acc_reg_comp_idx=row_idx % 2,
                kernal_size=0,  # 3x3 kernel
                line_buffer_row_shift=1,
                line_buffer_idx=row_idx % 2,
                is_padding_col=1,  # Enable padding
                weight_parall_mode=0,
                is_new=0 if row_idx == 0 else 1,
                transnum=9,  # 3x3 = 9 weights
                bas_addr=layer_idx * 100,  # Offset per layer
                is_bilinear_bicubic=0,
                offset_reg_idx=0
            )
            
            # Store results
            DataStorer.dispatch(
                quant_config_idx=layer_idx % 2,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=0,
                pooling_out_mode=0,
                pooling_out_new=0,
                is_pooling=0,
                reg_out_idx=row_idx % 2,
                acc_mode=0,
                transfer_num=1,
                store_mode=0,
                stride=32,
                base_addr_pooling=0,
                base_addrs_res=row_idx * 32,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx='b' if layer_idx % 2 == 0 else 'a'
            )
    
    # ========================================
    # Step 3: Store results off-chip
    # ========================================
    logger.info("  [3/4] Off-chip storing...")
    
    OffchipDataStorer.dispatch(
        src_buffer='b',  # Output from last layer
        transnum=1024,
        base_addr=0
    )
    
    # ========================================
    # Step 4: Get instructions
    # ========================================
    instructions = Inst.code_list
    logger.info(f"  [4/4] Generated {len(instructions)} instructions")
    
    return instructions


def save_instructions(instructions: list, output_dir: str):
    """Save instructions to text and binary files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text format
    inst_file = f"{output_dir}/usr_net_inst.txt"
    with open(inst_file, 'w') as f:
        for inst in instructions:
            f.write(str(inst) + '\n')
    
    logger.info(f"Saved {len(instructions)} instructions to {inst_file}")
    
    # Try to assemble to binary
    try:
        from assembler import compile_file
        
        binary_file = f"{output_dir}/usr_net_inst.bin"
        compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
        
        binary_size = os.path.getsize(binary_file)
        logger.info(f"Assembled binary: {binary_file} ({binary_size} bytes)")
        
        return {
            'text': inst_file,
            'binary': binary_file,
            'text_size': os.path.getsize(inst_file),
            'binary_size': binary_size
        }
    except Exception as e:
        logger.warning(f"Binary assembly failed: {e}")
        return {
            'text': inst_file,
            'binary': None,
            'text_size': os.path.getsize(inst_file)
        }


def main():
    """Main compilation flow."""
    print("=" * 70)
    print("USRNet: Frontend IR → Backend Instructions")
    print("=" * 70)
    print()
    
    # Configuration
    OUTPUT_DIR = "./output/usr_net"
    
    # Step 1: Analyze existing IR (from frontend)
    print("Step 1: Analyzing existing IR...")
    model_info = analyze_existing_ir(OUTPUT_DIR)
    
    if model_info is None:
        print("\n❌ Error: No IR files found!")
        print("\nPlease run the frontend compilation first:")
        print("  python examples/compile_usr_net.py")
        return
    
    print("✓ IR analysis complete\n")
    
    # Step 2: Generate instructions (backend)
    print("Step 2: Generating hardware instructions...")
    instructions = generate_instructions_for_usrnet(model_info)
    print("✓ Instruction generation complete\n")
    
    # Step 3: Save outputs
    print("Step 3: Saving instructions...")
    outputs = save_instructions(instructions, OUTPUT_DIR)
    print("✓ Instructions saved\n")
    
    # Summary
    print("=" * 70)
    print("Compilation Summary")
    print("=" * 70)
    print(f"Instructions: {len(instructions)}")
    print(f"Text file:    {outputs['text']}")
    if outputs['binary']:
        print(f"Binary file:  {outputs['binary']}")
    print()
    
    print("✓ Compilation complete!")
    print()
    print("Next steps:")
    print("  1. Review instructions: cat", outputs['text'])
    print("  2. Check instruction count and verify correctness")
    print("  3. For full USRNet, adapt sd_sr_codegen.py patterns")
    print()


if __name__ == '__main__':
    main()

