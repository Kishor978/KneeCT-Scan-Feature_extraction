import os
import sys
import torch
import numpy as np
from datetime import datetime
import argparse
from src import bone_segmentation_task,densenet_pipeline

class Task3Pipeline:
    """Complete Task III pipeline manager."""
    
    def __init__(self, input_path, output_base_dir, config=None):
        self.input_path = input_path
        self.output_base_dir = output_base_dir
        self.config = config or self._default_config()
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"task3_results_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Task III Pipeline initialized")
        print(f"Input: {input_path}")
        print(f"Output: {self.output_dir}")
    
    def _default_config(self):
        """Default configuration parameters."""
        return {
            # Segmentation parameters
            'bone_threshold': 200,
            'min_component_size': 10000,
            'morph_iterations': 2,
            'target_size': (64, 64, 64),
            
            # CNN parameters
            'use_pretrained': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Output parameters
            'save_intermediate': True,
            'visualize': True
        }
    
    def run_complete_pipeline(self):
        """Execute the complete Task III pipeline."""
        print("\n" + "="*60)
        print("STARTING TASK III COMPLETE PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Segmentation
            print("\n📋 STEP 1: SEGMENTATION-BASED SPLITTING")
            print("-" * 40)
            regions, masks, metadata = self._run_segmentation()
           
            print(f"Regions extracted: {list(regions.keys())}")
            # Step 2: 3D CNN Feature Extraction  
            print("\n🧠 STEP 2: 3D CNN FEATURE EXTRACTION")
            print("-" * 40)
            similarity_dict,results_df = self._run_feature_extraction(regions)
            
            # Step 3: Results Analysis and Export
            print("\n📊 STEP 3: RESULTS ANALYSIS")
            print("-" * 40)
            self._analyze_and_export_results(results_df, regions, masks, metadata)
            
            print("\n✅ TASK III PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved to: {self.output_dir}")
            
            return results_df, similarity_dict, regions, masks
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            raise
    
    def _run_segmentation(self):
        """Run segmentation step."""
        print("Running bone segmentation...")
        
        # Create segmentation subdirectory
        seg_output_dir = os.path.join(self.output_dir, "segmentation")
        
        # Run segmentation with Task III optimizations
        regions, masks, metadata = bone_segmentation_task(
            input_path=self.input_path,
            output_dir=seg_output_dir,
            bone_threshold=self.config['bone_threshold'],
            min_component_size=self.config['min_component_size'],
            morph_iterations=self.config['morph_iterations'],
            target_size=self.config['target_size'],
            visualize=self.config['visualize']
        )
        
        # Move tensors to appropriate device
        device = torch.device(self.config['device'])
        for region_name in regions:
            regions[region_name] = regions[region_name].to(device)
        
        print(f"✅ Segmentation completed")
        print(f"   - Regions: {list(regions.keys())}")
        print(f"   - Tensor shapes: {[regions[k].shape for k in regions.keys()]}")
        print(f"   - Device: {device}")
        
        return regions, masks, metadata
    
    def _run_feature_extraction(self, regions):
        """Run 3D CNN feature extraction step."""
        print("Converting 2D DenseNet121 to 3D and extracting features...")
        
        # Create feature extraction subdirectory
        features_output_dir = os.path.join(self.output_dir, "features")
        print(regions)
        for key, value in regions.items():
            print(f"Key: {key}, Type of value: {type(value)}, Value: {value}")

        # Run 3D CNN pipeline
        similarity_dict,results_df = densenet_pipeline(
            regions_dict=regions,           
        )
        
        print(f"✅ Feature extraction completed")
        print(f"   - Model: 3D DenseNet121 (converted from 2D pretrained)")
        
        return similarity_dict,results_df
    
    def _analyze_and_export_results(self, results_df, regions, masks, metadata):
        """Analyze results and create comprehensive exports."""
        print("Analyzing results and creating exports...")
        
        # 1. Save main CSV result (as required by Task III)
        main_csv_path = os.path.join(self.output_dir, "task3_cosine_similarities.csv")
        results_df.to_csv(main_csv_path, index=False)
        
        # 2. Create detailed analysis report
        self._create_analysis_report(results_df, regions, masks, metadata)
        
        # 3. Save configuration used
        self._save_pipeline_config()
        
        # 4. Create README for results
        self._create_results_readme()
        
        print(f"✅ Results analysis completed")
        print(f"   - Main CSV: {main_csv_path}")
        print(f"   - Detailed report: {os.path.join(self.output_dir, 'analysis_report.txt')}")
    
    def _create_analysis_report(self, results_df, regions, masks, metadata):
        """Create detailed analysis report."""
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("TASK III ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Pipeline info
            f.write("PIPELINE INFORMATION:\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_path}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Data info
            f.write("DATA INFORMATION:\n")
            f.write(f"Original shape: {metadata['original_shape']}\n")
            f.write(f"Target shape: {metadata['target_shape']}\n")
            f.write(f"Device used: {self.config['device']}\n\n")
            
            # Segmentation stats
            f.write("SEGMENTATION STATISTICS:\n")
            total_voxels = np.prod(metadata['original_shape'])
            femur_voxels = np.sum(masks['femur'])
            tibia_voxels = np.sum(masks['tibia'])
            background_voxels = total_voxels - femur_voxels - tibia_voxels
            
            f.write(f"Total voxels: {total_voxels:,}\n")
            f.write(f"Femur voxels: {femur_voxels:,} ({100*femur_voxels/total_voxels:.2f}%)\n")
            f.write(f"Tibia voxels: {tibia_voxels:,} ({100*tibia_voxels/total_voxels:.2f}%)\n")
            f.write(f"Background voxels: {background_voxels:,} ({100*background_voxels/total_voxels:.2f}%)\n\n")
            
            # Feature extraction results
            f.write("COSINE SIMILARITY RESULTS:\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\n")
            
            # Interpretation
            f.write("INTERPRETATION:\n")
            similarity_cols = [col for col in results_df.columns if 'vs' in col]
            for col in similarity_cols:
                similarity = results_df[col].iloc[0]
                f.write(f"{col}: {similarity:.4f} - ")
                if similarity > 0.8:
                    f.write("High similarity\n")
                elif similarity > 0.5:
                    f.write("Moderate similarity\n")
                else:
                    f.write("Low similarity\n")
    
    def _save_pipeline_config(self):
        """Save pipeline configuration."""
        config_path = os.path.join(self.output_dir, "pipeline_config.txt")
        
        with open(config_path, 'w') as f:
            f.write("TASK III PIPELINE CONFIGURATION\n")
            f.write("="*40 + "\n\n")
            
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
    
    def _create_results_readme(self):
        """Create README file explaining the results."""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        with open(readme_path, 'w') as f:
            f.write("# Task III: Knee CT Feature Extraction Pipeline\n")

def main():
    """Main entry point without command line interface."""
    
    # === Replace these with your desired file paths and settings ===
    input_path = "data/3702_left_knee.nii.gz"  # Path to your input CT scan
    output_dir = "results/task3"          # Base output directory
    bone_threshold = 200                  # Segmentation threshold
    target_size = (64, 64, 64)            # CNN input target size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualize = False                      # Enable visualization
    
    # Validate input file
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Setup configuration
    config = {
        'bone_threshold': bone_threshold,
        'min_component_size': 10000,
        'morph_iterations': 2,
        'target_size': target_size,
        'use_pretrained': True,
        'device': device,
        'save_intermediate': True,
        'visualize': visualize
    }
    
    # Initialize and run pipeline
    pipeline = Task3Pipeline(
        input_path=input_path,
        output_base_dir=output_dir,
        config=config
    )
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()
