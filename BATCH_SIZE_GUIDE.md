# Batch Size Support Documentation

## Overview
The video processing system now supports 6 specific batch sizes: 1, 4, 9, 16, 25, and 36 frames per batch.

## Batch Size Options

### batch_size=1 (1x1 grid)
- **Processing**: Single frame at a time
- **Speed**: Slowest
- **Accuracy**: Highest (each frame processed individually)
- **Memory**: Lowest
- **Use case**: High-precision scenarios, low-end hardware

### batch_size=4 (2x2 grid) - Default
- **Processing**: 4 frames arranged in 2x2 grid
- **Speed**: Good balance
- **Accuracy**: High
- **Memory**: Moderate
- **Use case**: Most general scenarios, recommended default

### batch_size=9 (3x3 grid)
- **Processing**: 9 frames arranged in 3x3 grid
- **Speed**: Faster
- **Accuracy**: Good
- **Memory**: Higher
- **Use case**: Longer videos, better hardware

### batch_size=16 (4x4 grid)
- **Processing**: 16 frames arranged in 4x4 grid
- **Speed**: Very fast
- **Accuracy**: Good (may miss rapid changes)
- **Memory**: High
- **Use case**: Long videos, powerful hardware

### batch_size=25 (5x5 grid)
- **Processing**: 25 frames arranged in 5x5 grid
- **Speed**: Very fast
- **Accuracy**: Good for stable scenes
- **Memory**: Very high
- **Use case**: High throughput processing, very long videos

### batch_size=36 (6x6 grid)
- **Processing**: 36 frames arranged in 6x6 grid
- **Speed**: Fastest
- **Accuracy**: Good for slow-changing scenes
- **Memory**: Highest
- **Use case**: Maximum throughput, very powerful hardware

## Usage Examples

### Command Line
```bash
# Use default batch size (4)
python facial_emotion_analysis.py --video input.mp4

# Use specific batch size
python facial_emotion_analysis.py --video input.mp4 --batch_size 9

# Use large batch for high throughput
python facial_emotion_analysis.py --video input.mp4 --batch_size 25

# Maximum batch size for fastest processing
python facial_emotion_analysis.py --video input.mp4 --batch_size 36

# Combine with sampling
python facial_emotion_analysis.py --video input.mp4 --batch_size 16 --sample_step 3
```

### Interactive Mode
The system will prompt for batch size selection when running interactively:
```
Batch size options:
1 - Single frame (1x1 grid) - Slowest but most accurate
4 - Small batch (2x2 grid) - Good balance (default)
9 - Medium batch (3x3 grid) - Faster processing
16 - Large batch (4x4 grid) - Fast processing
25 - Extra large batch (5x5 grid) - High throughput
36 - Maximum batch (6x6 grid) - Fastest processing
Choose batch size (1/4/9/16/25/36) or press Enter for default:
```

## Performance Characteristics

| Batch Size | Grid | Relative Speed | Memory Usage | Best For |
|------------|------|----------------|--------------|----------|
| 1 | 1x1 | 1x (baseline) | Low | Precision tasks |
| 4 | 2x2 | 2-3x faster | Medium | General use |
| 9 | 3x3 | 4-6x faster | High | Long videos |
| 16 | 4x4 | 6-8x faster | Very High | Bulk processing |
| 25 | 5x5 | 8-12x faster | Very High | High throughput |
| 36 | 6x6 | 10-15x faster | Extreme | Maximum speed |

## Technical Details

The batch processing works by:
1. Arranging multiple frames into a grid image
2. Running face detection on the entire grid
3. Mapping detected faces back to individual frames
4. Applying emotion recognition to each detected face
5. Distributing results to all frames (including unprocessed ones via inheritance)

This approach significantly reduces the number of ONNX model inference calls while maintaining good accuracy.
