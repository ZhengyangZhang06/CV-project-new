# CV Project Visualization Tools Guide

This is a comprehensive guide for the visualization tools in an ONNX-based face detection and emotion recognition project. This guide provides detailed instructions on how to use various visualization tools and their features.

## Project Overview

This project implements a complete facial emotion analysis pipeline, including:
- ONNX-based face detection models
- ResNet18 emotion recognition model
- Batch processing optimization algorithms
- Multiple visualization tools

## Visualization Tools Overview

### 1. Batch Detection Visualization (`batch_detection_visualization.py`)

**Main Features:**
- Demonstrates the complete batch face detection workflow
- Visualizes the transformation from individual frames to grid composition
- Shows face detection and emotion recognition results

**Supported Batch Sizes:** 1, 2, 4, 6, 9, 16, 25, 36

#### Basic Usage

```bash
# Process video with default model
python batch_detection_visualization.py --video j1_480P_30fps.mp4

# Specify face detection model and batch size
python batch_detection_visualization.py --video j1_480P_30fps.mp4 --face_model models/version-RFB-320.onnx --batch_size 9

# Custom output directory and frame offset
python batch_detection_visualization.py --video j1_480P_30fps.mp4 --output custom_output --frame_offset 100 --batch_size 16
```

#### Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--video` | string | Required | Input video file path |
| `--face_model` | string | Optional | ONNX face detection model path |
| `--batch_size` | int | 4 | Number of frames to process (1,2,4,6,9,16,25,36) |
| `--frame_offset` | int | 0 | Number of frames to skip from start |
| `--output` | string | "output" | Output directory |

#### Output Files

Generated visualization files are saved in `output/{model_name}/` directory:

1. **`original_frames_batch{N}.png`** - Original input frames
2. **`batch_grid_{R}x{C}.png`** - Batch grid image (with grid boundaries)
3. **`resized_grid_batch{N}.png`** - Resized image for model input
4. **`grid_detection_{R}x{C}.png`** - Face detection results on grid
5. **`emotion_detection_batch{N}.png`** - Individual frames with emotion labels

Where:
- `{N}` = batch size
- `{R}x{C}` = grid rows and columns
- `{model_name}` = automatically detected model name

### 2. Grid Visualization Details (`utils/grid_visualization.py`)

**Main Features:**
- Detailed visualization of grid transformation process
- Display frame dimensions and model input information
- Provides grid layout optimization algorithms

#### Basic Usage

```bash
# Generate grid details visualization
python -m utils.grid_visualization --video j1_480P_30fps.mp4 --face_model models/version-RFB-320.onnx --output grid_details.png
```

#### Core Functions

**Grid Size Optimization:**
```python
# Automatically select optimal grid layout
grid_rows, grid_cols = get_optimal_grid_dimensions(batch_size, frame_width, frame_height)
```

**Supported Grid Configurations:**
- 1 frame: 1×1
- 2 frames: 1×2 or 2×1 (based on frame aspect ratio)
- 4 frames: 2×2
- 6 frames: 2×3 or 3×2 (based on frame aspect ratio)
- 9 frames: 3×3
- 16 frames: 4×4
- 25 frames: 5×5
- 36 frames: 6×6

### 3. Main Program Analysis (`facial_emotion_analysis.py`)

**Main Features:**
- Complete video emotion analysis pipeline
- Real-time processing and display
- Audio merging functionality

#### Basic Usage

```bash
# Interactive mode
python facial_emotion_analysis.py

# Command line mode
python facial_emotion_analysis.py --video input.mp4 --output output.mp4 --sample_rate 15

# Process all frames
python facial_emotion_analysis.py --video input.mp4 --output output.mp4 --process_all

# No display during processing
python facial_emotion_analysis.py --video input.mp4 --output output.mp4 --no_display
```

#### Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--video` | string | Optional | Input video file path |
| `--output` | string | Optional | Output video file path |
| `--sample_rate` | int | 15 | Frames per second to process |
| `--process_all` | flag | False | Process all frames instead of sampling |
| `--no_display` | flag | False | Disable real-time display |
| `--face_model` | string | Optional | ONNX face detection model path |
| `--no_audio` | flag | False | Do not merge original video audio |

## Supported Models

### Face Detection Models

The project supports multiple ONNX face detection models:

1. **RFB-320** (`version-RFB-320.onnx`)
   - Input size: 320×240
   - Features: Balanced speed and accuracy

2. **RFB-640** (`version-RFB-640.onnx`)
   - Input size: 640×480
   - Features: High accuracy detection

3. **Slim-320** (`version-slim-320.onnx`)
   - Input size: 320×240
   - Features: Lightweight model

4. **Simplified** (`version-RFB-320_simplified.onnx`)
   - Input size: 320×240
   - Features: Optimized version

### Emotion Recognition Model

- **Model:** ResNet18
- **Training Data:** FER2013
- **Supported Emotions:** 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Input Size:** 48×48 grayscale images

## Advanced Features

### 1. Automatic Model Recognition

The system can automatically identify model types based on model filename and input dimensions:

```python
model_name = get_model_name_from_path_and_size(face_model_path, input_width, input_height)
# Output examples: "rfb_320", "slim_320", "rfb_640"
```

### 2. Batch Processing Optimization

Uses grid stitching to improve processing efficiency:

```python
# 4-frame batch processing example
frames = [frame1, frame2, frame3, frame4]
grid_image = create_batch_grid(frames, 2, 2)  # 2×2 grid
```

### 3. Coordinate Mapping

Maps grid detection results back to original frames:

```python
frame_idx, local_bbox = map_face_to_frame(face_box, grid_cols, frame_width, frame_height)
```

## Performance Statistics

### Batch Processing Effects

Performance comparison for different batch sizes:

| Batch Size | Grid Layout | Recommended Use Case |
|------------|-------------|---------------------|
| 1 | 1×1 | Single frame analysis |
| 4 | 2×2 | Standard batch processing |
| 9 | 3×3 | Medium batch processing |
| 16 | 4×4 | Large batch processing |
| 25 | 5×5 | Extra large batch processing |

### Model Performance Comparison

| Model | Input Size | Detection Accuracy | Processing Speed |
|-------|------------|-------------------|------------------|
| RFB-640 | 640×480 | High | Medium |
| RFB-320 | 320×240 | Medium | Fast |
| Slim-320 | 320×240 | Medium | Fastest |

## Output Examples

### Visualization Results Display

1. **Original Frame Grid**
   - Shows input consecutive frames
   - Frame dimension information

2. **Batch Processing Grid**
   - Stitched grid image
   - Grid boundary lines
   - Size annotations

3. **Face Detection Results**
   - Detection boxes and confidence scores
   - Grid coordinate system

4. **Emotion Recognition Results**
   - Emotion labels and probabilities
   - Colored detection boxes
   - Frame statistics

### Sample Log Output

```
Processing 9 frames in batch mode
Detected model type: rfb_320
Video frame range: 0 to 8 (out of 7200 total frames)
Using 3x3 grid for 9 frames
Model: rfb_320 (320x240)
Total faces detected: 12
Average faces per frame: 1.3
Visualization complete! All images saved to output/rfb_320/
```

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: Model file 'models/version-RFB-320.onnx' not found.
   ```
   - Check model file path
   - Ensure ONNX model exists

2. **Cannot open video file**
   ```
   Error: Cannot open video file 'input.mp4'
   ```
   - Check video file format
   - Ensure file path is correct

3. **Out of memory**
   - Reduce batch size
   - Lower video resolution

### Performance Optimization Recommendations

1. **Choose appropriate batch size**
   - Sufficient GPU memory: Use 16 or 25
   - Limited GPU memory: Use 4 or 9
   - CPU processing: Use 1 or 4

2. **Choose appropriate model**
   - High accuracy requirement: RFB-640
   - Balanced performance: RFB-320
   - Fast processing: Slim-320

3. **Adjust sampling rate**
   - Real-time processing: 15-30 FPS
   - Offline analysis: 10-15 FPS
   - Quick preview: 5-10 FPS

## Technical Details

### Data Flow

1. **Video Input** → **Frame Extraction** → **Batch Organization**
2. **Grid Stitching** → **Size Adjustment** → **ONNX Inference**
3. **Face Detection** → **Coordinate Mapping** → **Emotion Recognition**
4. **Result Visualization** → **File Saving**

### Dependencies

- **OpenCV**: Image processing and video I/O
- **PyTorch**: Emotion recognition model
- **ONNX Runtime**: Face detection inference
- **Matplotlib**: Visualization plotting
- **NumPy**: Numerical computation
- **tqdm**: Progress bar display

## Update Log

### Latest Version Features

- ✅ Support for flexible batch processing (1-36 frames)
- ✅ Automatic model recognition and directory organization
- ✅ Optimized grid layout algorithms
- ✅ Detailed performance statistics
- ✅ Complete visualization workflow

### Future Plans

- 🔄 Support for more ONNX model formats
- 🔄 Add real-time video stream processing
- 🔄 Integration of more emotion recognition models
- 🔄 Optimize large video file processing

---

**Contact Information:** For questions or suggestions, please refer to the project documentation or submit an issue.

**Version:** v2.0 (May 2025)
