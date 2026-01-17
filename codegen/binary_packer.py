"""
Package instructions and weights into binary format.
"""

import sys
sys.path.append('..')

from assembler import compile_file
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class BinaryPacker:
    """Package compiled output into binary format."""
    
    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
    
    def pack(self, instructions: List[Dict], params: Dict, output_name: str = 'model'):
        """
        Package instructions and parameters.
        
        Args:
            instructions: List of instruction dictionaries
            params: Model parameters
            output_name: Base name for output files
        """
        # Write instructions to file
        inst_file = f"{self.output_dir}/{output_name}_inst.txt"
        with open(inst_file, 'w') as f:
            for inst in instructions:
                f.write(str(inst) + '\n')
        
        logger.info(f"Wrote {len(instructions)} instructions to {inst_file}")
        
        # Assemble to binary
        binary_file = f"{self.output_dir}/{output_name}_inst.bin"
        try:
            compile_file(inst_file, binary_file, split=False, pad_and_cut=True)
            logger.info(f"Assembled binary to {binary_file}")
        except Exception as e:
            logger.error(f"Assembly failed: {e}")
        
        # Save parameters (simplified)
        param_file = f"{self.output_dir}/{output_name}_params.json"
        param_info = {name: {'shape': list(p.shape), 'dtype': str(p.dtype)} 
                     for name, p in params.items()}
        with open(param_file, 'w') as f:
            json.dump(param_info, f, indent=2)
        
        logger.info(f"Saved parameter info to {param_file}")
        
        return {
            'instructions': inst_file,
            'binary': binary_file,
            'params': param_file
        }

