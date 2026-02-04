#!/usr/bin/env python3
"""
Compile FSRCNN with aten::pad support.

This script demonstrates how to compile PyTorch models that use
torch.nn.functional.pad by applying a monkey patch to TVM.
"""

import sys
sys.path.append('..')

# CRITICAL: Apply patch BEFORE importing TVM
from patches.pytorch_pad_support import patch_pytorch_frontend
print("="*70)
print("Applying aten::pad patch to TVM PyTorch frontend...")
print("="*70)
patch_result = patch_pytorch_frontend()

if not patch_result:
    print("\n✗ Failed to apply patch! Cannot continue.")
    sys.exit(1)

print()

# Now safe to import TVM
import logging
from compiler.driver import TVMVISCompiler
from compiler.config import CompilerConfig, PassConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main compilation flow."""
    print("="*70)
    print("Compiling FSRCNN Model")
    print("="*70)
    print()
    
    # Configuration
    MODEL_PATH = "../models/fsrcnn.pth"  # Adjust path as needed
    OUTPUT_DIR = "./output/fsrcnn"
    
    # Create pass configuration
    pass_config = PassConfig(
        opt_level=3,                      # Maximum optimization
        enable_fold_constant=True,
        enable_fold_scale_axis=True
    )
    
    # Create compiler configuration
    config = CompilerConfig(
        target='vis_vpu',
        pass_config=pass_config,
        output_dir=OUTPUT_DIR,
        visualize_passes=False
    )
    
    compiler = TVMVISCompiler(config)
    
    # Compile model
    try:
        logger.info(f"Compiling model: {MODEL_PATH}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("")
        
        result = compiler.compile(
            MODEL_PATH,
            model_format='pytorch',  # or 'onnx' if you have .onnx file
        )
        
        print()
        print("="*70)
        print("✓ Compilation Successful!")
        print("="*70)
        print(f"Instructions:     {result['num_instructions']}")
        print(f"Parameters:       {result['num_parameters']}")
        print(f"Output files in:  {OUTPUT_DIR}")
        print()
        
        return result
        
    except Exception as e:
        print()
        print("="*70)
        print("✗ Compilation Failed!")
        print("="*70)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

