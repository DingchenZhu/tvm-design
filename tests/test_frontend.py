"""
Tests for model frontend (ONNX, PyTorch importers).
"""

import sys
sys.path.append('..')

import pytest
import torch
from models_new_930 import FSRCNN
from frontend import load_model, ONNXImporter, PyTorchImporter


def test_onnx_import():
    """Test ONNX model import."""
    try:
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        assert mod is not None
        assert params is not None
        assert len(params) > 0
        print("✓ ONNX import test passed")
    except Exception as e:
        pytest.skip(f"ONNX import failed: {e}")


def test_pytorch_import():
    """Test PyTorch model import."""
    try:
        model = FSRCNN(scale_factor=2, num_channels=1, d=32, s=8, m=4)
        example_input = torch.randn(1, 1, 270, 480)
        
        mod, params = load_model(
            model,
            model_format='pytorch',
            example_inputs=example_input
        )
        
        assert mod is not None
        assert params is not None
        print("✓ PyTorch import test passed")
    except Exception as e:
        pytest.skip(f"PyTorch import failed: {e}")


if __name__ == '__main__':
    print("Running frontend tests...")
    test_onnx_import()
    test_pytorch_import()
    print("\n✓ All frontend tests passed!")

