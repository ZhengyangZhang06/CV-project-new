# Batch Face Detection and Emotion Analysis Visualization Tool

This toolkit is designed to visualize the batch face detection and emotion recognition process based on ONNX. These tools help you understand the image processing workflow in batch processing and how batch results are mapped back to individual images.

## Main Features

1. **Batch Face Detection Visualization** - Demonstrates how 4 images are processed simultaneously for face detection
2. **Batch Grid Transformation** - Shows the transformation process from individual images to a 2x2 grid and model input
3. **Results Visualization** - Renders face detection and emotion analysis results on the original images

## Usage Instructions

### Batch Detection Visualization

This script demonstrates the complete batch detection and emotion analysis process:

```
python batch_detection_visualization.py --video <video_file> --face_model <ONNX_model_path>
```

Optional parameters:
- `--num_frames` - Number of frames to process (maximum 4, default 4)
- `--frame_offset` - Number of frames to skip from the beginning (default 0)
- `--output` - Directory to save visualization results

### Grid Details Visualization

This script focuses on visualizing detailed information about the grid transformation process:

```
python -m utils.grid_visualization --video <video_file> --face_model <ONNX_model_path> --output grid_details.png
```

## Examples

```
# Batch detection visualization with default model
python batch_detection_visualization.py --video j1_480P_30fps.mp4

# Using a specific model and custom output directory
python batch_detection_visualization.py --video j1_480P_30fps.mp4 --face_model models/version-RFB-320.onnx --output custom_output

# Visualizing only the grid processing details
python -m utils.grid_visualization --video j1_480P_30fps.mp4 --face_model models/version-RFB-320.onnx
```

## Output Files Description

Batch processing visualization generates the following image files (saved to the `output/` directory by default):

1. `output/original_frames.png` - Original input frames
2. `output/batch_grid.png` - 2x2 batch grid image
3. `output/resized_grid.png` - Resized grid (used as model input)
4. `output/grid_detection.png` - Face detection results on the batch grid
5. `output/emotion_detection.png` - Individual frames with emotion labels
6. `output/grid_details.png` - Transformation process visualization with detailed dimension information

## Notes

- This tool is for visualization only and does not modify the original processing pipeline
- Batch processing uses a 2x2 grid to process up to 4 frames simultaneously
- The tool automatically detects and uses the input dimensions required by the model
