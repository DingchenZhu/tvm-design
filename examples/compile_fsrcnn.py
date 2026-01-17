#!/usr/bin/env python3
"""
Example script to compile FSRCNN (PyTorch) model.
"""

import sys
sys.path.append('..')

import logging
import torch
from models_new_930 import FSRCNN
from compiler import TVMVISCompiler, CompilerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Compile FSRCNN model."""
    print("="*60)
    print("Compiling FSRCNN (PyTorch) Model")
    print("="*60)
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 1, 270, 480)
    
    # Configuration
    config = CompilerConfig(
        output_dir='./output/fsrcnn',
        output_name='fsrcnn',
        dump_ir=True,
        dump_instructions=True,
    )
    
    # Create compiler
    compiler = TVMVISCompiler(config=config)
    
    # Compile model
    try:
        result = compiler.compile(
            model=model,
            model_format='pytorch',
            example_inputs=example_input
        )
        
        print("\n" + "="*60)
        print("Compilation Results:")
        print("="*60)
        print(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Instructions: {result['num_instructions']}")
        print(f"Parameters: {result['num_parameters']}")
        print(f"\nOutput files:")
        for key, path in result['output_files'].items():
            print(f"  {key}: {path}")
        
    except Exception as e:
        print(f"\nCompilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

