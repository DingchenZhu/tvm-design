"""
Pass pipeline manager for orchestrating optimization passes.

Provides preset optimization pipelines for different optimization levels
and target hardware.
"""

import tvm
from tvm import relay
from tvm.relay import transform
from typing import List, Optional, Dict
import logging

from .base_pass import RelayPass, SequentialPass, visualize_pass_effect
from .basic_passes import (
    InferTypePass,
    FoldConstantPass,
    SimplifyInferencePass,
    SimplifyExprPass,
    FoldScaleAxisPass,
    CanonicalizeCastPass,
    CanonicalizeOpsPass,
)
from .fusion_pass import FusionPass, CustomFusionPass, create_hardware_aware_fusion_pass
from .dce_pass import DeadCodeEliminationPass, EnhancedDCEPass
from .redundancy_elimination import RedundancyEliminationPass

logger = logging.getLogger(__name__)


class PassPipeline:
    """
    Manages a pipeline of optimization passes.
    """
    
    def __init__(self, name: str, opt_level: int = 2):
        """
        Initialize pass pipeline.
        
        Args:
            name: Pipeline name
            opt_level: Optimization level (0-3)
        """
        self.name = name
        self.opt_level = opt_level
        self.passes: List[RelayPass] = []
    
    def add_pass(self, pass_obj: RelayPass):
        """
        Add a pass to the pipeline.
        
        Args:
            pass_obj: Pass to add
        """
        self.passes.append(pass_obj)
        logger.debug(f"Added pass: {pass_obj.name}")
    
    def remove_pass(self, pass_name: str):
        """
        Remove a pass from the pipeline.
        
        Args:
            pass_name: Name of pass to remove
        """
        self.passes = [p for p in self.passes if p.name != pass_name]
    
    def get_pass(self, pass_name: str) -> Optional[RelayPass]:
        """
        Get a pass by name.
        
        Args:
            pass_name: Name of the pass
            
        Returns:
            Pass object if found, None otherwise
        """
        for p in self.passes:
            if p.name == pass_name:
                return p
        return None
    
    def apply(self, mod: tvm.IRModule, visualize: bool = False) -> tvm.IRModule:
        """
        Apply all passes in the pipeline.
        
        Args:
            mod: Input IR module
            visualize: Whether to visualize the effect of each pass
            
        Returns:
            Optimized IR module
        """
        logger.info(f"Applying pass pipeline: {self.name} ({len(self.passes)} passes)")
        
        prev_mod = mod
        
        for i, pass_obj in enumerate(self.passes):
            if not pass_obj.enabled:
                logger.info(f"  [{i+1}/{len(self.passes)}] Skipping disabled pass: {pass_obj.name}")
                continue
            
            logger.info(f"  [{i+1}/{len(self.passes)}] Applying: {pass_obj.name}")
            
            try:
                mod = pass_obj(mod)
                
                if visualize:
                    visualize_pass_effect(prev_mod, mod, pass_obj.name)
                    prev_mod = mod
                
            except Exception as e:
                logger.error(f"  ✗ Pass {pass_obj.name} failed: {e}")
                raise
        
        logger.info(f"Pipeline {self.name} completed successfully")
        return mod
    
    def __call__(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Shortcut to apply the pipeline."""
        return self.apply(mod)


def create_basic_pipeline(opt_level: int = 2) -> PassPipeline:
    """
    Create a basic optimization pipeline.
    
    Args:
        opt_level: Optimization level
        
    Returns:
        Pass pipeline
    """
    pipeline = PassPipeline("Basic", opt_level=opt_level)
    
    pipeline.add_pass(InferTypePass())
    pipeline.add_pass(SimplifyInferencePass())
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(SimplifyExprPass())
    pipeline.add_pass(InferTypePass())
    
    return pipeline


def create_standard_pipeline(opt_level: int = 2) -> PassPipeline:
    """
    Create a standard optimization pipeline with fusion and DCE.
    
    Args:
        opt_level: Optimization level
        
    Returns:
        Pass pipeline
    """
    pipeline = PassPipeline("Standard", opt_level=opt_level)
    
    # Initial cleanup
    pipeline.add_pass(InferTypePass())
    pipeline.add_pass(SimplifyInferencePass())
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(SimplifyExprPass())
    
    # Canonicalization
    pipeline.add_pass(CanonicalizeCastPass())
    pipeline.add_pass(CanonicalizeOpsPass())
    
    # Scale folding (important for BN fusion)
    pipeline.add_pass(FoldScaleAxisPass())
    
    # Redundancy elimination
    pipeline.add_pass(RedundancyEliminationPass(apply_cse=True))
    
    # Operator fusion
    pipeline.add_pass(FusionPass(fuse_opt_level=min(opt_level, 3)))
    
    # Dead code elimination
    pipeline.add_pass(DeadCodeEliminationPass())
    
    # Final cleanup
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(InferTypePass())
    
    return pipeline


def create_aggressive_pipeline(opt_level: int = 3) -> PassPipeline:
    """
    Create an aggressive optimization pipeline with all optimizations.
    
    Args:
        opt_level: Optimization level
        
    Returns:
        Pass pipeline
    """
    pipeline = PassPipeline("Aggressive", opt_level=opt_level)
    
    # Initial cleanup
    pipeline.add_pass(InferTypePass())
    pipeline.add_pass(SimplifyInferencePass())
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(SimplifyExprPass())
    
    # First round of optimizations
    pipeline.add_pass(CanonicalizeCastPass())
    pipeline.add_pass(CanonicalizeOpsPass())
    pipeline.add_pass(FoldScaleAxisPass())
    pipeline.add_pass(RedundancyEliminationPass(apply_cse=True))
    pipeline.add_pass(FoldConstantPass())
    
    # Operator fusion (custom + built-in)
    pipeline.add_pass(CustomFusionPass(
        patterns=None,  # Use all patterns
        enable_tvm_fusion=True,
        fuse_opt_level=3
    ))
    
    # Post-fusion cleanup
    pipeline.add_pass(SimplifyExprPass())
    pipeline.add_pass(EnhancedDCEPass())
    
    # Second round of optimizations
    pipeline.add_pass(RedundancyEliminationPass(apply_cse=True))
    pipeline.add_pass(FoldConstantPass())
    
    # Final cleanup
    pipeline.add_pass(DeadCodeEliminationPass())
    pipeline.add_pass(InferTypePass())
    
    return pipeline


def create_visvpu_pipeline(opt_level: int = 3) -> PassPipeline:
    """
    Create optimization pipeline optimized for VIS VPU hardware.
    
    Args:
        opt_level: Optimization level
        
    Returns:
        Pass pipeline
    """
    pipeline = PassPipeline("VIS_VPU", opt_level=opt_level)
    
    # Initial cleanup
    pipeline.add_pass(InferTypePass())
    pipeline.add_pass(SimplifyInferencePass())
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(SimplifyExprPass())
    
    # Canonicalization and cleanup
    pipeline.add_pass(CanonicalizeCastPass())
    pipeline.add_pass(CanonicalizeOpsPass())
    pipeline.add_pass(FoldScaleAxisPass())
    
    # VIS VPU-specific optimizations
    pipeline.add_pass(RedundancyEliminationPass(apply_cse=True))
    
    # Hardware-aware fusion
    pipeline.add_pass(create_hardware_aware_fusion_pass(
        target='vis_vpu',
        opt_level=opt_level
    ))
    
    # Post-fusion cleanup
    pipeline.add_pass(SimplifyExprPass())
    pipeline.add_pass(EnhancedDCEPass())
    
    # Final passes
    pipeline.add_pass(FoldConstantPass())
    pipeline.add_pass(InferTypePass())
    
    return pipeline


class PassManager:
    """
    High-level pass manager with preset pipelines.
    """
    
    PRESETS = {
        'basic': create_basic_pipeline,
        'standard': create_standard_pipeline,
        'aggressive': create_aggressive_pipeline,
        'vis_vpu': create_visvpu_pipeline,
    }
    
    @classmethod
    def create_pipeline(
        cls,
        preset: str = 'standard',
        opt_level: int = 2,
        custom_passes: Optional[List[RelayPass]] = None
    ) -> PassPipeline:
        """
        Create a pass pipeline from preset or custom passes.
        
        Args:
            preset: Preset name ('basic', 'standard', 'aggressive', 'vis_vpu')
            opt_level: Optimization level
            custom_passes: Optional list of custom passes to add
            
        Returns:
            Pass pipeline
        """
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        
        # Create preset pipeline
        pipeline = cls.PRESETS[preset](opt_level=opt_level)
        
        # Add custom passes if provided
        if custom_passes:
            for pass_obj in custom_passes:
                pipeline.add_pass(pass_obj)
        
        return pipeline
    
    @classmethod
    def optimize(
        cls,
        mod: tvm.IRModule,
        preset: str = 'standard',
        opt_level: int = 2,
        visualize: bool = False
    ) -> tvm.IRModule:
        """
        Convenience method to optimize a module.
        
        Args:
            mod: Input IR module
            preset: Pipeline preset
            opt_level: Optimization level
            visualize: Whether to visualize pass effects
            
        Returns:
            Optimized IR module
        """
        pipeline = cls.create_pipeline(preset, opt_level)
        return pipeline.apply(mod, visualize=visualize)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing pass pipeline manager...")
    
    try:
        from frontend import load_model
        from .base_pass import count_ops
        
        # Load model
        print("\n" + "="*60)
        print("Loading model...")
        mod, params = load_model('../USR_Net.onnx', model_format='onnx')
        
        # Count ops before
        ops_before = count_ops(mod)
        print(f"\nOps before optimization: {sum(ops_before.values())}")
        
        # Test different presets
        for preset in ['basic', 'standard', 'aggressive']:
            print("\n" + "="*60)
            print(f"Testing {preset} pipeline...")
            
            mod_opt = PassManager.optimize(mod, preset=preset, opt_level=2)
            
            ops_after = count_ops(mod_opt)
            reduction = sum(ops_before.values()) - sum(ops_after.values())
            print(f"  Ops after: {sum(ops_after.values())} (reduction: {reduction})")
        
        print("\n✓ Pass pipeline manager test complete!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

