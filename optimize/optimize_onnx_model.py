"""
Script to optimize ONNX models by removing initializers from inputs.
This addresses the warning: "Initializer appears in graph inputs and will not be treated as constant value/weight"
"""

import argparse
import os
import onnx
from onnx import helper, shape_inference
import numpy as np

def remove_initializers_from_input(model_path, output_path=None):
    """Remove initializers from model inputs to enable graph optimizations like const folding."""
    if output_path is None:
        base_name = os.path.basename(model_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(model_path), f"{name}_optimized{ext}")
    
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    
    # Get all initializer names
    initializer_names = {init.name for init in model.graph.initializer}
    
    # Create new inputs without the initializers
    new_inputs = []
    removed_inputs = []
    
    for input in model.graph.input:
        if input.name not in initializer_names:
            new_inputs.append(input)
        else:
            removed_inputs.append(input.name)
    
    # Replace inputs with new list
    if removed_inputs:
        del model.graph.input[:]
        model.graph.input.extend(new_inputs)
        print(f"Removed {len(removed_inputs)} initializers from graph inputs:")
        for name in removed_inputs[:10]:  # Print first 10 for brevity
            print(f"  - {name}")
        if len(removed_inputs) > 10:
            print(f"  - ... and {len(removed_inputs) - 10} more")
    else:
        print("No initializers found in inputs. Model already optimized.")
        return model_path
    
    # Run shape inference to ensure the model is valid
    try:
        inferred_model = shape_inference.infer_shapes(model)
        model = inferred_model
        print("Shape inference completed successfully.")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing with the modified model.")
    
    # Save the optimized model
    print(f"Saving optimized model to {output_path}...")
    onnx.save(model, output_path)
    print(f"Model optimization complete. Original size: {os.path.getsize(model_path):,} bytes, "
          f"Optimized size: {os.path.getsize(output_path):,} bytes")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX models by removing initializers from inputs")
    parser.add_argument("--input", required=True, help="Path to input ONNX model")
    parser.add_argument("--output", help="Path to save optimized model (default: input_optimized.onnx)")
    args = parser.parse_args()
    
    optimized_path = remove_initializers_from_input(args.input, args.output)
    print(f"Optimized model saved to: {optimized_path}")

if __name__ == "__main__":
    main()
