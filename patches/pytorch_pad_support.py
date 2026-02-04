"""
Monkey patch to add aten::pad support to TVM PyTorch frontend.

Usage:
    from patches.pytorch_pad_support import patch_pytorch_frontend
    patch_pytorch_frontend()  # Call BEFORE importing TVM
    
    import tvm
    from tvm import relay
    # Now you can import models with pad operations
"""

def patch_pytorch_frontend():
    """
    Add aten::pad operator support to TVM PyTorch frontend.
    
    This patches the PyTorchOpConverter class at runtime without
    modifying TVM source files.
    
    Returns:
        bool: True if patch successful, False otherwise
    """
    try:
        from tvm.relay.frontend import pytorch
        from tvm import relay
        from tvm.relay import op as _op
        
        # Get the converter class
        converter_class = pytorch.PyTorchOpConverter
        
        # Define the _pad method
        def _pad(self, inputs, input_types):
            """
            Implement aten::pad operator.
            
            PyTorch signature:
                torch.nn.functional.pad(input, pad, mode='constant', value=0)
            
            Args:
                inputs: List of inputs
                    [0]: data tensor
                    [1]: pad tuple/list [left, right, top, bottom, ...]
                    [2]: mode ('constant', 'reflect', 'replicate', 'circular')
                    [3]: value (for constant padding)
                input_types: Input type information
            
            Returns:
                Relay expression for padded tensor
            """
            data = inputs[0]
            
            # Extract padding list
            if isinstance(inputs[1], list):
                pad_list = inputs[1]
            else:
                try:
                    pad_list = self.infer_shape(inputs[1])
                except:
                    # If infer_shape fails, try to extract from constant
                    if isinstance(inputs[1], relay.expr.Constant):
                        import numpy as np
                        pad_list = inputs[1].data.numpy().tolist()
                    else:
                        raise ValueError(f"Cannot extract padding from {type(inputs[1])}")
            
            # Get mode (default: 'constant')
            mode = 'constant'
            if len(inputs) > 2 and inputs[2] is not None:
                if isinstance(inputs[2], str):
                    mode = inputs[2]
                elif isinstance(inputs[2], relay.expr.Constant):
                    mode = inputs[2].data.numpy().item()
            
            # Get fill value (default: 0)
            pad_value = 0.0
            if len(inputs) > 3 and inputs[3] is not None:
                if isinstance(pad_value, (int, float)):
                    pad_value = float(inputs[3])
                elif isinstance(inputs[3], relay.expr.Constant):
                    import numpy as np
                    pad_value = float(inputs[3].data.numpy().item())
            
            # Get input shape
            data_shape = self.infer_shape(data)
            ndim = len(data_shape)
            
            # Initialize pad_width with zeros (no padding for all dimensions)
            pad_width = [[0, 0]] * ndim
            
            # PyTorch pad format: [left, right, top, bottom, front, back]
            # Applied to last dimensions first
            # 
            # For 4D tensor (N, C, H, W):
            #   pad=(1,2,3,4) means W_pad=[1,2], H_pad=[3,4]
            # 
            # Relay format: [[N_before, N_after], [C_before, C_after], ...]
            
            if len(pad_list) >= 2:
                # Pad last dimension (Width for 4D, or last dim in general)
                pad_width[-1] = [int(pad_list[0]), int(pad_list[1])]
            
            if len(pad_list) >= 4:
                # Pad second-to-last dimension (Height for 4D)
                pad_width[-2] = [int(pad_list[2]), int(pad_list[3])]
            
            if len(pad_list) >= 6:
                # Pad third-to-last dimension (Depth/Channel)
                pad_width[-3] = [int(pad_list[4]), int(pad_list[5])]
            
            if len(pad_list) >= 8:
                # Pad fourth-to-last dimension (Batch)
                pad_width[-4] = [int(pad_list[6]), int(pad_list[7])]
            
            # Convert PyTorch mode to Relay mode
            mode_map = {
                'constant': 'constant',
                'reflect': 'reflect',
                'replicate': 'edge',      # Relay uses 'edge' for replicate
                'circular': 'wrap'
            }
            relay_mode = mode_map.get(mode, 'constant')
            
            # Create Relay pad operation
            if relay_mode == 'constant':
                result = _op.nn.pad(data, pad_width=pad_width, pad_value=pad_value)
            else:
                result = _op.nn.pad(data, pad_width=pad_width, pad_mode=relay_mode)
            
            return result
        
        # Attach the method to the class
        converter_class._pad = _pad
        
        # Patch __init__ to add to convert_map
        original_init = converter_class.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)
            
            # Add pad to convert_map
            self.convert_map["aten::pad"] = self._pad
        
        converter_class.__init__ = patched_init
        
        print("✓ TVM PyTorch frontend patched with aten::pad support")
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch PyTorch frontend: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pad():
    """Test the pad implementation."""
    import torch
    import tvm
    from tvm import relay
    
    print("\n" + "="*70)
    print("Testing aten::pad Implementation")
    print("="*70 + "\n")
    
    # Create a simple model with padding
    class PadTestModel(torch.nn.Module):
        def forward(self, x):
            # Test constant padding
            return torch.nn.functional.pad(x, (1, 1, 2, 2), 'constant', 0)
    
    model = PadTestModel()
    model.eval()
    
    # Create example input
    x = torch.randn(1, 3, 10, 10)
    
    # Get expected output from PyTorch
    with torch.no_grad():
        expected = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {expected.shape}")
    print(f"Expected:     {tuple(expected.shape)}")
    
    # Trace the model
    try:
        scripted = torch.jit.trace(model, x)
        print("✓ Model traced successfully")
    except Exception as e:
        print(f"✗ Tracing failed: {e}")
        return False
    
    # Convert to Relay
    try:
        mod, params = relay.frontend.from_pytorch(
            scripted,
            [("input", (1, 3, 10, 10))]
        )
        print("✓ Pad operator successfully converted to Relay!")
        print(f"\nGenerated Relay IR:")
        print("-" * 70)
        print(mod)
        print("-" * 70)
        return True
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Apply the patch
    success = patch_pytorch_frontend()
    
    if success:
        # Test it
        test_pad()

