import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project modules
from models.resnet import ResNet18
from utils.face_detection import load_face_model, predict_faces
from utils.grid_visualization import (
    get_optimal_grid_dimensions, visualize_grid, create_batch_grid, 
    add_grid_lines, map_face_to_frame, save_visualization
)
from utils.batch_visual_func import (
		extract_video_frames, process_face_detection, process_emotions_for_frames,get_model_name_from_path_and_size
)

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_emotion_model(model_path='models/fer2013_resnet_best.pth'):
    """Load the emotion recognition model
    
    Args:
        model_path: Path to the model file
    
    Returns:
        torch.nn.Module: Loaded model or None if failed
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return None
    
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded emotion recognition model from {model_path}")
    return model

def visualize_batch_detection(video_path, face_model_path=None, num_frames=4, frame_offset=0, output_path=None):
    """Visualize batch face detection on frames from a video
    
    Args:
        video_path: Path to input video
        face_model_path: Path to ONNX face detection model
        num_frames: Number of frames to process (supports 1, 2, 4, 6, 9, 16, 25, 36)
        frame_offset: Skip this many frames from the start
        output_path: Path to save visualization image
    """
    # Validate and adjust number of frames
    supported_batch_sizes = [1, 2, 4, 6, 9, 16, 25, 36]
    if num_frames not in supported_batch_sizes:
        # Find the closest supported batch size
        closest = min(supported_batch_sizes, key=lambda x: abs(x - num_frames))
        print(f"Warning: {num_frames} frames not directly supported. Using {closest} frames instead.")
        num_frames = closest
    
    print(f"Processing {num_frames} frames in batch mode")
    
    # Load models
    model = load_emotion_model()
    if not model:
        return
    
    result = load_face_model(face_model_path)
    if not result:
        return
    ort_session, input_width, input_height = result
    input_name = ort_session.get_inputs()[0].name
    print(f"Model expected input size: {input_width}x{input_height}")
    
    # Determine model name and create model-specific output directory
    model_name = get_model_name_from_path_and_size(face_model_path, input_width, input_height)
    print(f"Detected model type: {model_name}")
      # Extract frames from video
    print(f"\n--- Frame Extraction ---")
    frames, video_props = extract_video_frames(video_path, num_frames, frame_offset)
    if not frames:
        return
    
    frame_width = video_props['width']
    frame_height = video_props['height']
    total_frames = video_props.get('total_frames', 0)
    
    print(f"Video frame range: {frame_offset} to {frame_offset + num_frames - 1} (out of {total_frames} total frames)")
    print(f"Extracted {len(frames)} consecutive frames for batch processing")
    
    # Create model-specific output directory
    if not output_path:
        output_path = "output"
    
    # Add model name to output path
    model_output_path = os.path.join(output_path, model_name)
    os.makedirs(model_output_path, exist_ok=True)
    print(f"Saving results to: {model_output_path}")
    
    # Get optimal grid dimensions based on frame aspect ratio
    grid_rows, grid_cols = get_optimal_grid_dimensions(num_frames, frame_width, frame_height)
    print(f"Using {grid_rows}x{grid_cols} grid for {num_frames} frames")
      # Visualize original frames
    print(f"Visualizing {len(frames)} original frames...")
    fig_original = visualize_grid(frames, titles=[f"Original {i+1}" for i in range(len(frames))],
                                 suptitle=f"Original Frames (Batch Size: {num_frames})")
    save_visualization(fig_original, model_output_path, f"original_frames_batch{num_frames}.png")
    plt.close()
    
    # Create batch grid for processing
    grid_image = create_batch_grid(frames, grid_rows, grid_cols)
    
    # Visualize the grid image
    print(f"Batch grid size: {grid_image.shape[1]}x{grid_image.shape[0]}")
    fig_grid = plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{grid_rows}x{grid_cols} Batch Grid ({grid_image.shape[1]}x{grid_image.shape[0]}) - {num_frames} frames")
    plt.axis('off')
    
    # Add grid lines to show frame boundaries
    add_grid_lines(plt.gca(), grid_rows, grid_cols, frame_height, frame_width)
    save_visualization(fig_grid, model_output_path, f"batch_grid_{grid_rows}x{grid_cols}.png")
    plt.close()
    
    # Process face detection
    face_boxes, face_probs, processed_grid = process_face_detection(
        grid_image, ort_session, input_name, input_width, input_height)
    
    # Visualize the resized grid
    fig_resized = plt.figure(figsize=(10, 10))
    plt.imshow(processed_grid)
    plt.title(f"Resized Grid for ONNX Model ({input_width}x{input_height})")
    plt.axis('off')
    save_visualization(fig_resized, model_output_path, f"resized_grid_batch{num_frames}.png")
    plt.close()
    
    # Draw face detections on grid
    grid_with_faces = grid_image.copy()
    for i in range(len(face_boxes)):
        x1, y1, x2, y2 = face_boxes[i]
        cv2.rectangle(grid_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 3)
        conf_text = f"Conf: {face_probs[i]:.2f}"
        cv2.putText(grid_with_faces, conf_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Visualize grid with face detections
    fig_detection = plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(grid_with_faces, cv2.COLOR_BGR2RGB))
    plt.title(f"Face Detection on {grid_rows}x{grid_cols} Grid (Found {len(face_boxes)} faces)")
    plt.axis('off')
    add_grid_lines(plt.gca(), grid_rows, grid_cols, frame_height, frame_width, alpha=0.5)
    save_visualization(fig_detection, model_output_path, f"grid_detection_{grid_rows}x{grid_cols}.png")
    plt.close()
    
    # Map faces back to individual frames
    frame_faces = [[] for _ in range(len(frames))]
    
    for i in range(len(face_boxes)):
        frame_idx, local_bbox = map_face_to_frame(face_boxes[i], grid_cols, frame_width, frame_height)
        
        if frame_idx < len(frames):
            x1, y1, x2, y2 = local_bbox
            w, h = x2 - x1, y2 - y1
            
            # Skip if face region is too small
            if w > 0 and h > 0:
                frame_faces[frame_idx].append({
                    'bbox': local_bbox,
                    'confidence': float(face_probs[i])
                })
    
    # Process emotions for each frame
    frames_with_emotion = process_emotions_for_frames(frames, frame_faces, model)
    
    # Visualize frames with emotion detection
    fig_emotion = visualize_grid(frames_with_emotion, 
                                titles=[f"Frame {i+1} with emotion" for i in range(len(frames_with_emotion))],
                                suptitle=f"Emotion Detection Results (Batch Size: {num_frames})")
    save_visualization(fig_emotion, model_output_path, f"emotion_detection_batch{num_frames}.png")
    plt.close()
    
    # Print summary
    total_faces = sum(len(faces) for faces in frame_faces)
    print(f"\n--- Batch Processing Summary ---")
    print(f"Model: {model_name} ({input_width}x{input_height})")
    print(f"Batch size: {num_frames} frames")
    print(f"Grid layout: {grid_rows}x{grid_cols}")
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per frame: {total_faces/len(frames):.1f}")
    print(f"Visualization complete! All images saved to {model_output_path}/")
    print(f"Generated files:")
    print(f"  - original_frames_batch{num_frames}.png")
    print(f"  - batch_grid_{grid_rows}x{grid_cols}.png")
    print(f"  - resized_grid_batch{num_frames}.png")
    print(f"  - grid_detection_{grid_rows}x{grid_cols}.png")
    print(f"  - emotion_detection_batch{num_frames}.png")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize batch face detection process')
    parser.add_argument('--video', required=True, help='Path to the input video file')
    parser.add_argument('--face_model', help='Path to the ONNX face detection model file')
    parser.add_argument('--batch_size', type=int, default=4, choices=[1, 2, 4, 6, 9, 16, 25, 36], 
                       help='Number of frames to process in batch (1, 2, 4, 6, 9, 16, 25, or 36)')
    parser.add_argument('--frame_offset', type=int, default=0, help='Skip this many frames from the start')
    parser.add_argument('--output', default='output', help='Directory to save visualization images')
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return
    
    # Run visualization
    visualize_batch_detection(
        video_path=args.video,
        face_model_path=args.face_model,
        num_frames=args.batch_size,
        frame_offset=args.frame_offset,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
