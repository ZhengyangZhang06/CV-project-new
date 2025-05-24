# 🎭 AI-Powered Facial Emotion Analysis System

A comprehensive real-time facial emotion recognition system that combines ONNX-based face detection with PyTorch emotion classification for video analysis.

## 🌟 Features

- **🎯 Real-time Emotion Recognition**: Detect 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **⚡ Optimized Batch Processing**: Process 1-36 frames simultaneously using intelligent grid layouts
- **🔧 Multiple ONNX Models**: Support for RFB-320, RFB-640, Slim-320 (These 3 model are pretrained, not by us), and optimized variants
- **📊 Advanced Visualization**: Comprehensive visualization tools for analysis and debugging
- **🎵 Audio Integration**: Preserve original video audio in processed outputs
- **📈 Performance Analytics**: Detailed processing statistics and performance monitoring
- **🛠️ Model Optimization**: ONNX model optimization tools for better performance

## 🏗️ Architecture

```
📁 CV-project-new/
├── 🎬 facial_emotion_analysis.py     # Main video processing pipeline
├── 📊 batch_detection_visualization.py # Batch visualization tool
├── 📚 models/                        # Model files
│   ├── fer2013_resnet_best.pth      # Emotion recognition model
│   ├── resnet.py                    # ResNet18 architecture
│   └── *.onnx                       # Face detection models
├── 🔧 utils/                        # Core utilities
│   ├── video_processing.py          # Video processing engine
│   ├── face_detection.py            # ONNX face detection
│   ├── emotion_recognition.py       # Emotion classification
│   ├── grid_visualization.py        # Grid layout algorithms
│   └── batch_visual_func.py         # Batch processing functions
├── ⚡ optimize/                     # Model optimization tools
│   ├── optimize_onnx_model.py       # Single model optimizer
│   └── optimize_all_models.py       # Batch optimizer
├── 📂 output/                       # Generated visualizations
├── 📖 past_work/                    # Previous implementations
└── 📄 Documentation/                # Guides and documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install torch torchvision opencv-python onnxruntime matplotlib numpy tqdm pillow
```

### Basic Usage

1. **Interactive Mode** (Recommended for beginners):
```bash
python facial_emotion_analysis.py
```

2. **Command Line Mode**:
```bash
# Basic processing
python facial_emotion_analysis.py --video input.mp4 --output output.mp4

# Advanced options
python facial_emotion_analysis.py \
    --video input.mp4 \
    --output output.mp4 \
    --batch_size 9 \
    --sample_step 2 \
    --face_model models/version-RFB-320.onnx
```

3. **Batch Visualization**:
```bash
# Visualize batch processing workflow
python batch_detection_visualization.py \
    --video j1_480P_30fps.mp4 \
    --batch_size 16 \
    --output custom_output \
    --frame_offset 100
```

## 📊 Batch Processing System

### Supported Batch Sizes

| Batch Size | Grid Layout | Speed | Accuracy | Use Case |
|------------|-------------|-------|----------|----------|
| 1 | 1×1 | Slowest | Highest | High precision |
| 4 | 2×2 | Balanced | High | **Default/Recommended** |
| 9 | 3×3 | Fast | Good | Longer videos |
| 16 | 4×4 | Very Fast | Good | High throughput |
| 25 | 5×5 | Very Fast | Moderate | Long videos |
| 36 | 6×6 | Fastest | Moderate | Maximum speed |

### Performance Benchmarks

*Tested on 30fps video (1004×480, 2min18s) on laptop CPU:*

**RFB-640 Model:**
- Batch Size 1: 70s (highest accuracy)
- Batch Size 4: 33s 
- Batch Size 9: 25s
- Batch Size 16: 23s (best balance) ⭐
- Batch Size 25: 17s (face detection may be less reliable, but still good)

**RFB-320 Model:**
- Batch Size 1: 37s
- Batch Size 4: 30s ⭐
- Batch Size 9: 25s (detection reliability starts to decrease)

## 🎯 Available Models

### Face Detection Models (ONNX)

1. **RFB-320** (`version-RFB-320.onnx`)
   - Input: 320×240
   - Best for: Balanced speed and accuracy

2. **RFB-640** (`version-RFB-640.onnx`)
   - Input: 640×480
   - Best for: Maximum accuracy

3. **Slim-320** (`version-slim-320.onnx`)
   - Input: 320×240
   - Best for: Lightweight processing

4. **Simplified** (`version-RFB-320_simplified.onnx`)
   - Input: 320×240
   - Best for: Optimized performance

### Emotion Recognition Model

- **Architecture**: ResNet18
- **Training Data**: FER2013 dataset
- **Input**: 48×48 grayscale images
- **Output**: 7 emotion classes with confidence scores

## 🔧 Advanced Configuration

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--video` | string | Interactive | Input video file path |
| `--output` | string | Interactive | Output video file path |
| `--batch_size` | int | 4 | Frames per batch (1,4,9,16,25,36) |
| `--sample_step` | int | 2 | Process every N frames (1-5) |
| `--face_model` | string | Auto | ONNX face detection model |
| `--process_all` | flag | False | Process all frames |
| `--no_display` | flag | False | Disable real-time preview |
| `--no_audio` | flag | False | Skip audio merging |

### Visualization Options

```bash
# Batch processing visualization
python batch_detection_visualization.py \
    --video input.mp4 \
    --batch_size 9 \
    --frame_offset 100 \
    --output visualization_output
```

**Generated Visualizations:**
- `original_frames_batch{N}.png` - Input frames
- `batch_grid_{R}x{C}.png` - Grid composition
- `resized_grid_batch{N}.png` - Model input
- `grid_detection_{R}x{C}.png` - Face detection results
- `emotion_detection_batch{N}.png` - Final emotion analysis

## ⚡ Model Optimization

### ONNX Model Optimization

Optimize your ONNX models for better performance:

```bash
# Optimize single model
python optimize/optimize_onnx_model.py \
    --input models/version-RFB-320.onnx \
    --output models/version-RFB-320-optimized.onnx

# Optimize all models (creates backups)
python optimize/optimize_all_models.py
```

**Benefits:**
- Eliminates ONNX Runtime warnings
- Improves inference speed
- Reduces memory usage
- Enables additional optimizations

## 📈 Performance Tuning

### Recommended Settings

**For Real-time Processing:**
```bash
--batch_size 25 --sample_step 2 --face_model models/version-RFB-640.onnx
```

**For High Accuracy:**
```bash
--batch_size 2 --sample_step 1 --face_model models/version-RFB-640.onnx
```

**For Fast Batch Processing:**
```bash
--batch_size 25 --sample_step 3 --face_model models/version-RFB-640.onnx
```

### Sample Step Guidelines

- **sample_step=1**: Process every frame (slowest, highest accuracy)
- **sample_step=2**: Good balance (recommended) ⭐
- **sample_step=3**: Acceptable for most content
- **sample_step=4**: Fast processing, some emotion changes may be missed
- **sample_step=5**: Too fast, significant emotion loss

## 🛠️ Technical Details

### Processing Pipeline

1. **Video Input** → **Frame Extraction** → **Batch Organization**
2. **Grid Stitching** → **Resize for ONNX** → **Face Detection**
3. **Coordinate Mapping** → **Face Extraction** → **Emotion Recognition**
4. **Result Application** → **Video Output** → **Audio Merging**

### Dependencies

```python
torch>=1.9.0          # PyTorch for emotion model
torchvision>=0.10.0    # Image transformations
opencv-python>=4.5.0   # Video processing
onnxruntime>=1.8.0     # ONNX model inference
matplotlib>=3.3.0      # Visualization
numpy>=1.19.0          # Numerical computing
tqdm>=4.60.0          # Progress bars
pillow>=8.0.0         # Image processing
```

## 🎨 Visualization System

The project includes comprehensive visualization tools:

### Grid Layout Algorithm
- Automatically calculates optimal grid dimensions
- Preserves aspect ratios
- Minimizes padding and distortion

### Emotion Color Coding
- **Angry**: Red (255,0,0)
- **Disgust**: Purple (128,0,128)
- **Fear**: Blue-Purple (128,0,255)
- **Happy**: Green (0,255,0)
- **Sad**: Blue (0,0,255)
- **Surprise**: Yellow (255,255,0)
- **Neutral**: Light Gray (192,192,192)

## 📚 Project Evolution

### Past Work (`past_work/`)
- **dlib-based implementations**: Earlier versions using dlib for face detection
- **Training experiments**: Model training and validation scripts for emotion recognition
- **Feature extraction**: SIFT and keypoint-based approaches

### Current System Advantages
- ✅ 3-5x faster processing with batch optimization
- ✅ Better accuracy with ONNX face detection models
- ✅ More robust emotion recognition
- ✅ Comprehensive visualization and debugging tools
- ✅ Production-ready pipeline with error handling

## 🔍 Troubleshooting

### Common Issues

**1. Model File Not Found**
```
Error: Model file 'models/fer2013_resnet_best.pth' not found.
```
- Ensure the emotion model file exists in the `models/` directory

**2. ONNX Runtime Warnings**
```
Initializer appears in graph inputs and will not be treated as constant value
```
- Use the optimization tools: `python optimize/optimize_all_models.py`

**3. Low Detection Accuracy**
- Try smaller batch sizes (4 or 1)
- Use higher resolution models (RFB-640)
- Reduce sample_step to 1 or 2

**4. Out of Memory**
- Reduce batch_size to 4 or 1
- Use smaller input models (RFB-320 or Slim-320)

### Performance Tips

- **CPU Processing**: Use batch_size=4, RFB-320 model
- **GPU Available**: Can handle larger batch sizes (16-25)
- **Long Videos**: Use sample_step=2-3 to reduce processing time
- **Short Clips**: Use sample_step=1 for maximum detail

## 📄 Documentation

- 📖 **[Batch Size Guide](BATCH_SIZE_GUIDE.md)**: Detailed batch processing documentation
- 🎨 **[Visualization Guide](Visualization_Guide.md)**: Complete visualization tools guide
- ⚡ **[Optimization Guide](optimize/onnx_optimization.md)**: ONNX model optimization instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with multiple video formats
4. Submit a pull request with detailed description

## 📊 Sample Usage Examples

### Example 1: Quick Emotion Analysis
```bash
# Process video with default settings
python facial_emotion_analysis.py --video family_video.mp4
```

### Example 2: High-Performance Batch Processing
```bash
# Fast processing for long videos
python facial_emotion_analysis.py \
    --video long_presentation.mp4 \
    --output analyzed_presentation.mp4 \
    --batch_size 16 \
    --sample_step 3 \
    --no_display
```

### Example 3: Detailed Analysis with Visualization
```bash
# Create detailed visualizations
python batch_detection_visualization.py \
    --video test_clip.mp4 \
    --batch_size 9 \
    --face_model models/version-RFB-640.onnx \
    --output detailed_analysis
```

## 🏆 Performance Stats

*Real-world performance on various hardware configurations:*

| Hardware | Model | Batch Size | Speed | Quality |
|----------|-------|------------|-------|---------|
| Laptop CPU | RFB-640 | 16 | 20s/2min video | Good ⭐ |

---

**Version**: 2.0 (May 2025)  
**License**: MIT  
**Author**: ChangxunPan, Zhengyang Zhang

For questions, issues, or feature requests, please refer to the project documentation or create an issue in the repository.
