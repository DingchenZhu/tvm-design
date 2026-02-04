"""
Main compiler driver for TVM VIS VPU compiler stack.
"""

import os
import sys
sys.path.append('..')

import tvm
from tvm import relay
import torch
import torch.nn as nn
from typing import Union, Dict, Optional, Tuple, Any
import logging

from frontend import ModelLoader
from passes import PassManager
from codegen.layer_mapper import LayerMapper
from codegen.binary_packer import BinaryPacker
from .config import CompilerConfig, DEFAULT_CONFIG
from .target import VISVPUTarget, create_tvm_target

logger = logging.getLogger(__name__)


class TVMVISCompiler:
    """
    Main compiler for converting models to VIS VPU instructions.
    
    Pipeline:
    1. Import model (PyTorch/ONNX) to Relay IR
    2. Apply optimization passes
    3. Lower to TIR (optional)
    4. Schedule and tile (optional)
    5. Generate VIS VPU instructions
    6. Assemble to binary
    """
    
    def __init__(self, config: CompilerConfig = None):
        """
        Initialize compiler.
        
        Args:
            config: Compiler configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.model_loader = ModelLoader()
        self.layer_mapper = LayerMapper()
        self.binary_packer = BinaryPacker(output_dir=self.config.output_dir)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Initialized TVM VIS Compiler (target={self.config.target})")
    
    def compile(
        self,
        model: Union[str, nn.Module],
        model_format: Optional[str] = None,
        example_inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compile model end-to-end.
        
        Args:
            model: Model path or PyTorch model object
            model_format: Model format ('pytorch', 'onnx', 'torchscript')
            example_inputs: Example inputs for PyTorch models
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with compilation results
        """
        logger.info("="*60)
        logger.info("Starting compilation")
        logger.info("="*60)
        
        # Stage 1: Import model
        logger.info("\n[Stage 1/5] Importing model...")
        mod, params = self._import_model(model, model_format, example_inputs, **kwargs)
        
        if self.config.dump_ir:
            self._save_ir(mod, "01_imported")
        
        # Stage 2: Optimize IR
        logger.info("\n[Stage 2/5] Optimizing IR...")
        mod = self._optimize(mod)
        
        if self.config.dump_ir:
            self._save_ir(mod, "02_optimized")
        
        # Stage 3: Lower to TIR (optional)
        logger.info("\n[Stage 3/5] Lowering to TIR...")
        if self.config.enable_scheduling:
            mod = self._lower_to_tir(mod, params)
            if self.config.dump_ir:
                self._save_ir(mod, "03_tir")
        
        # Stage 4: Generate instructions
        logger.info("\n[Stage 4/5] Generating instructions...")
        instructions = self._generate_instructions(mod, params)
        
        # Stage 5: Package output
        logger.info("\n[Stage 5/5] Packaging output...")
        output_files = self._package(instructions, params)
        
        logger.info("\n" + "="*60)
        logger.info("Compilation complete!")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("="*60)
        
        return {
            'success': True,
            'output_files': output_files,
            'num_instructions': len(instructions),
            'num_parameters': len(params),
        }
    
    def _import_model(self, model, model_format, example_inputs, **kwargs):
        """Import model to Relay IR."""
        if example_inputs is not None:
            mod, params = self.model_loader.load(
                model,
                model_format=model_format,
                example_inputs=example_inputs,
                **kwargs
            )
        else:
            mod, params = self.model_loader.load(
                model,
                model_format=model_format,
                **kwargs
            )
        
        logger.info(f"Imported model: {len(params)} parameters")
        return mod, params
    
    def _optimize(self, mod):
        """Apply optimization passes."""
        preset = 'vis_vpu' if self.config.target == 'vis_vpu' else 'standard'
        
        mod = PassManager.optimize(
            mod,
            preset=preset,
            opt_level=self.config.pass_config.opt_level,
            visualize=self.config.visualize_passes,
            enable_fold_constant=self.config.pass_config.enable_fold_constant,
            enable_fold_scale_axis=self.config.pass_config.enable_fold_scale_axis
        )
        
        return mod
    
    def _lower_to_tir(self, mod, params):
        """Lower Relay to TIR."""
        # This would involve calling relay.build with appropriate target
        # For now, we skip this for direct instruction generation
        logger.info("TIR lowering skipped (direct instruction generation)")
        return mod
    
    def _generate_instructions(self, mod, params):
        """Generate VIS VPU instructions from IR."""
        # Simplified: extract operators and generate instructions
        # In full implementation, would traverse IR and map each op
        
        logger.info("Generating instruction sequence...")
        instructions = self.layer_mapper.get_all_instructions()
        
        logger.info(f"Generated {len(instructions)} instructions")
        return instructions
    
    def _package(self, instructions, params):
        """Package instructions and parameters."""
        return self.binary_packer.pack(
            instructions,
            params,
            output_name=self.config.output_name
        )
    
    def _save_ir(self, mod, stage_name):
        """Save IR to file."""
        ir_file = f"{self.config.output_dir}/{stage_name}.txt"
        with open(ir_file, 'w') as f:
            f.write(mod.astext(show_meta_data=False))
        logger.info(f"Saved IR to {ir_file}")


def compile_model(
    model: Union[str, nn.Module],
    config: Optional[CompilerConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to compile a model.
    
    Args:
        model: Model path or object
        config: Compiler configuration
        **kwargs: Additional arguments for compilation
        
    Returns:
        Compilation results
    """
    compiler = TVMVISCompiler(config=config)
    return compiler.compile(model, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing compiler driver...")
    print("\nNote: Full test requires model files and TVM installation")
    print("This is a structure validation test.")
    
    # Test configuration
    config = CompilerConfig(
        output_dir='./test_output',
        output_name='test_model',
        dump_ir=True
    )
    
    compiler = TVMVISCompiler(config=config)
    print(f"\n✓ Compiler initialized with config: {config.target}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Optimization level: {config.pass_config.opt_level}")
    
    print("\n✓ Compiler driver structure validated!")

