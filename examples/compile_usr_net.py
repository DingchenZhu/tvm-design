#!/usr/bin/env python3
"""
Example script to compile USR_Net (ONNX) model.
"""

import sys
sys.path.append('..')

import logging
from compiler import TVMVISCompiler, CompilerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Compile USR_Net model."""
    print("="*60)
    print("Compiling USR_Net (ONNX) Model")
    print("="*60)
    
    # Configuration
    config = CompilerConfig(
        output_dir='./output/usr_net',
        output_name='usr_net',
        dump_ir=True,
        dump_instructions=True,
    )
    
    # Create compiler
    compiler = TVMVISCompiler(config=config)
    
    # Compile model
    try:
        result = compiler.compile(
            model='../USR_Net.onnx',
            model_format='onnx'
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

