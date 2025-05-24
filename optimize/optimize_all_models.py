"""
Script to optimize all ONNX models in the models directory
"""

import os
import glob
from optimize.optimize_onnx_model import remove_initializers_from_input

def optimize_all_models(directory="models", pattern="*.onnx", backup=True):
    """
    Optimize all ONNX models in the given directory that match the pattern
    
    Args:
        directory: Directory containing ONNX models
        pattern: Glob pattern to match model files
        backup: Whether to back up original models
    """
    # Find all ONNX models
    model_paths = glob.glob(os.path.join(directory, pattern))
    
    if not model_paths:
        print(f"No ONNX models found in {directory} matching pattern {pattern}")
        return
    
    print(f"Found {len(model_paths)} ONNX models to optimize")
    
    # Create backup directory if needed
    backup_dir = os.path.join(directory, "original")
    if backup and not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Created backup directory: {backup_dir}")
    
    # Process each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nProcessing model: {model_name}")
        
        if backup:
            # Copy original to backup folder
            import shutil
            backup_path = os.path.join(backup_dir, model_name)
            shutil.copy2(model_path, backup_path)
            print(f"Backed up original model to {backup_path}")
        
        # Optimize the model in place
        try:
            optimized_path = remove_initializers_from_input(model_path, model_path)
            print(f"Model optimized and saved to {optimized_path}")
        except Exception as e:
            print(f"Error optimizing model {model_name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize all ONNX models in a directory")
    parser.add_argument("--dir", default="models", help="Directory containing ONNX models")
    parser.add_argument("--pattern", default="*.onnx", help="Glob pattern to match model files")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups of original models")
    
    args = parser.parse_args()
    optimize_all_models(args.dir, args.pattern, not args.no_backup)
