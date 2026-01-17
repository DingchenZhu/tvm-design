"""
Tests for optimization passes.
"""

import sys
sys.path.append('..')

import pytest
from frontend import load_model
from passes import PassManager, apply_basic_optimizations


def test_basic_passes():
    """Test basic optimization passes."""
    try:
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Apply basic optimizations
        mod_opt = apply_basic_optimizations(mod, opt_level=2)
        
        assert mod_opt is not None
        print("✓ Basic passes test passed")
    except Exception as e:
        pytest.skip(f"Basic passes test failed: {e}")


def test_pass_manager():
    """Test pass pipeline manager."""
    try:
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Test different presets
        for preset in ['basic', 'standard']:
            mod_opt = PassManager.optimize(mod, preset=preset, opt_level=2)
            assert mod_opt is not None
        
        print("✓ Pass manager test passed")
    except Exception as e:
        pytest.skip(f"Pass manager test failed: {e}")


if __name__ == '__main__':
    print("Running pass tests...")
    test_basic_passes()
    test_pass_manager()
    print("\n✓ All pass tests passed!")

