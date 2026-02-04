"""
Patch for TVM PyTorch Frontend to support aten::pad operator.

This file contains the implementation that needs to be added to:
/home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py

Instructions:
1. Add the pad_generic function after the make_pad function (around line 1741)
2. Add the operator registration in the convert_map dictionary (around line 2950)
"""

# ============================================================================
# ADD THIS FUNCTION after make_pad() function (around line 1741)
# ============================================================================

def pad_generic(self, inputs, input_types):
    """
    Generic pad operator that supports multiple modes.

    PyTorch signature: pad(input, pad, mode='constant', value=0)

    Args:
        inputs[0]: input tensor
        inputs[1]: pad sizes (tuple/list)
        inputs[2]: mode string ('constant', 'reflect', 'replicate', 'circular')
        inputs[3]: value (for constant mode)
    """
    data = inputs[0]

    # Extract pad list
    if isinstance(inputs[1], list):
        pad_list = inputs[1]
    else:
        pad_list = list(self.infer_shape(inputs[1]))

    # Extract mode (default to 'constant')
    if len(inputs) > 2 and inputs[2] is not None:
        mode = inputs[2]
        # Convert string constant to actual string
        if hasattr(mode, 'data'):
            mode = str(mode.data.numpy().decode('utf-8'))
        elif isinstance(mode, str):
            mode = mode
        else:
            mode = 'constant'
    else:
        mode = 'constant'

    # Extract value (default to 0)
    pad_value = inputs[3] if len(inputs) > 3 and inputs[3] is not None else 0.0

    # Initialize paddings based on input length
    pad_len = len(self.infer_shape(data)) * 2
    paddings = [0] * pad_len

    # PyTorch pad format: (left, right, top, bottom, front, back)
    # TVM format: [(before, after), (before, after), ...]
    if len(pad_list) >= 2:
        paddings[-1] = pad_list[1]  # right
        paddings[-2] = pad_list[0]  # left
    if len(pad_list) >= 4:
        paddings[-3] = pad_list[3]  # bottom
        paddings[-4] = pad_list[2]  # top
    if len(pad_list) >= 6:
        paddings[-5] = pad_list[5]  # back
        paddings[-6] = pad_list[4]  # front

    # Group into tuple of 2 ints
    paddings = [paddings[i : i + 2] for i in range(0, len(paddings), 2)]

    # Convert to constants
    const_paddings = []
    for pad in paddings:
        const_paddings.append([])
        for p in pad:
            if not isinstance(p, int):
                from tvm.relay.frontend.common import _infer_value
                p = int(_infer_value(p, {}).numpy())
            const_paddings[-1].append(p)

    # Map PyTorch mode names to TVM mode names
    mode_map = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',
        'circular': 'wrap'  # TVM uses 'wrap' for circular padding
    }

    tvm_mode = mode_map.get(mode, 'constant')

    # Import relay op
    from tvm import relay
    _op = relay.op

    # Apply padding
    if tvm_mode == 'constant':
        return _op.nn.pad(data, const_paddings, pad_value=pad_value, pad_mode=tvm_mode)
    else:
        return _op.nn.pad(data, const_paddings, pad_mode=tvm_mode)


# ============================================================================
# ADD THIS LINE in the convert_map dictionary (around line 2950)
# ============================================================================

# Add this line after "aten::replication_pad3d": self.make_pad("edge"),
# "aten::pad": self.pad_generic,


# ============================================================================
# COMPLETE PATCH INSTRUCTIONS
# ============================================================================

PATCH_INSTRUCTIONS = """
To apply this patch manually:

1. Open /home/hansz/scratch-data/design/tvm/python/tvm/relay/frontend/pytorch.py

2. Find the make_pad function (around line 1703-1741)

3. Add the pad_generic function right after make_pad (after line 1741)

4. Find the convert_map dictionary (around line 2900-2950)

5. Add this line in the appropriate location:
   "aten::pad": self.pad_generic,

   Recommended location: after the other pad operators (around line 2950)

6. Save the file

7. Test by running: python compile_fsrcnn.py
"""

print(__doc__)
print(PATCH_INSTRUCTIONS)
