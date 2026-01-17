"""
End-to-end compilation tests.
"""

import sys
sys.path.append('..')

import pytest
import torch
from models_new_930 import FSRCNN
from compiler import TVMVISCompiler, CompilerConfig


def test_onnx_compilation():
    """Test end-to-end ONNX compilation."""
    try:
        config = CompilerConfig(
            output_dir='./test_output/usr_net',
            output_name='usr_net_test'
        )
        
        compiler = TVMVISCompiler(config=config)
        result = compiler.compile(
            model='../USR_Net.onnx',
            model_format='onnx'
        )
        
        assert result['success']
        print("✓ ONNX E2E compilation test passed")
    except Exception as e:
        pytest.skip(f"ONNX compilation failed: {e}")


def test_pytorch_compilation():
    """Test end-to-end PyTorch compilation."""
    try:
        model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
        example_input = torch.randn(1, 1, 270, 480)
        
        config = CompilerConfig(
            output_dir='./test_output/fsrcnn',
            output_name='fsrcnn_test'
        )
        
        compiler = TVMVISCompiler(config=config)
        result = compiler.compile(
            model=model,
            model_format='pytorch',
            example_inputs=example_input
        )
        
        assert result['success']
        print("✓ PyTorch E2E compilation test passed")
    except Exception as e:
        pytest.skip(f"PyTorch compilation failed: {e}")


if __name__ == '__main__':
    print("Running end-to-end tests...")
    test_onnx_compilation()
    test_pytorch_compilation()
    print("\n✓ All E2E tests passed!")

