import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project modules
from models.resnet import ResNet18
from utils.emotion_recognition import EMOTIONS, EMOTION_COLORS, preprocess_face, predict_emotion
from utils.face_detection import load_face_model, predict_faces

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_grid(frames, titles=None, grid_shape=(2, 2), figsize=(12, 8)):
    """Visualize a grid of frames with titles and dimensions"""
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        # Convert to RGB for matplotlib
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
        else:
            ax.imshow(frame, cmap='gray')
        
        # Add title with dimensions if available
        title = f"Frame {i+1}"
        if titles and i < len(titles):
            title = titles[i]
        h, w = frame.shape[:2]
        title += f" ({w}x{h})"
        ax.set_title(title)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def visualize_batch_detection(video_path, face_model_path=None, num_frames=4, frame_offset=0, output_path=None):
    """Visualize batch face detection on frames from a video
    
    Args:
        video_path: Path to input video
        face_model_path: Path to ONNX face detection model
        num_frames: Number of frames to process (max 4)
        frame_offset: Skip this many frames from the start
        output_path: Path to save visualization image
    """
    # Limit number of frames
    num_frames = min(num_frames, 4)
    
    # Load emotion recognition model
    model_path = 'models/fer2013_resnet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded emotion recognition model from {model_path}")
    
    # Load face detection model
    result = load_face_model(face_model_path)
    if not result:
        return
    
    # Unpack the model session and input dimensions
    ort_session, input_width, input_height = result
    input_name = ort_session.get_inputs()[0].name
    
    # Open video and extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS")
    print(f"Model expected input size: {input_width}x{input_height}")
    
    # Skip frames if needed
    for _ in range(frame_offset):
        ret = cap.read()
        if not ret:
            print("Error: Could not skip frames (video too short)")
            cap.release()
            return
    
    # Read frames
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        print(f"Warning: Could only read {len(frames)} frames")
      # Create output directory if needed
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = "output"
        os.makedirs(output_path, exist_ok=True)
    
    # Visualize original frames
    print(f"Visualizing {len(frames)} original frames...")
    fig_original = visualize_grid(frames, titles=[f"Original {i+1}" for i in range(len(frames))])
    plt.suptitle("Original Frames", fontsize=16)
    plt.savefig(os.path.join(output_path, "original_frames.png"))
    
    # Create batch grid for processing
    grid_h = 2 * frame_height
    grid_w = 2 * frame_width
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill the grid with frames
    for i, frame in enumerate(frames):
        row, col = divmod(i, 2)
        y_offset = row * frame_height
        x_offset = col * frame_width
        grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
      # Visualize the grid image
    print(f"Batch grid size: {grid_image.shape[1]}x{grid_image.shape[0]}")
    fig_grid = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.title(f"2x2 Batch Grid ({grid_image.shape[1]}x{grid_image.shape[0]})")
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "batch_grid.png"))
    
    # Preprocess the grid for inference
    grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    processed_grid = cv2.resize(grid_rgb, (input_width, input_height))
    
    # Visualize the resized grid
    fig_resized = plt.figure(figsize=(8, 8))
    plt.imshow(processed_grid)
    plt.title(f"Resized Grid for ONNX Model ({input_width}x{input_height})")
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "resized_grid.png"))
    
    # Complete preprocessing for inference
    image_mean = np.array([127, 127, 127])
    processed_grid_norm = (processed_grid - image_mean) / 128
    processed_grid_norm = np.transpose(processed_grid_norm, [2, 0, 1])
    processed_grid_norm = np.expand_dims(processed_grid_norm, axis=0)
    processed_grid_norm = processed_grid_norm.astype(np.float32)
    
    # Run inference
    print("Running face detection on batch grid...")
    confidences, boxes = ort_session.run(None, {input_name: processed_grid_norm})
    
    # Process face detection results
    face_boxes, _, face_probs = predict_faces(grid_w, grid_h, confidences, boxes, prob_threshold=0.6)
    
    # Create a copy of the grid image for visualization
    grid_with_faces = grid_image.copy()
    
    # Draw all face detections on the grid
    for i in range(len(face_boxes)):
        x1, y1, x2, y2 = face_boxes[i]
        cv2.rectangle(grid_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf_text = f"Conf: {face_probs[i]:.2f}"
        cv2.putText(grid_with_faces, conf_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
      # Visualize grid with face detections
    fig_detection = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(grid_with_faces, cv2.COLOR_BGR2RGB))
    plt.title(f"Face Detection on Grid (Found {len(face_boxes)} faces)")
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "grid_detection.png"))
    
    # Extract and process individual frames with detections
    frames_with_emotion = []
    
    # Process each detected face and map back to original frames
    frame_faces = [[] for _ in range(len(frames))]
    
    for i in range(len(face_boxes)):
        x1, y1, x2, y2 = face_boxes[i]
        
        # Determine which frame this face belongs to
        frame_row = int(y1 // frame_height)
        frame_col = int(x1 // frame_width)
        frame_idx = frame_row * 2 + frame_col
        
        if frame_idx >= len(frames):
            continue  # Skip faces in invalid frames
        
        # Adjust coordinates to the original frame
        x1_local = x1 - (frame_col * frame_width)
        y1_local = y1 - (frame_row * frame_height)
        x2_local = x2 - (frame_col * frame_width)
        y2_local = y2 - (frame_row * frame_height)
        
        # Make sure coordinates are within the frame bounds
        x1_local = max(0, min(x1_local, frame_width))
        y1_local = max(0, min(y1_local, frame_height))
        x2_local = max(0, min(x2_local, frame_width))
        y2_local = max(0, min(y2_local, frame_height))
        
        w_local = x2_local - x1_local
        h_local = y2_local - y1_local
        
        # Skip if face region is too small
        if w_local <= 0 or h_local <= 0:
            continue
        
        # Store face coordinates for this frame
        frame_faces[frame_idx].append({
            'bbox': (x1_local, y1_local, x2_local, y2_local),
            'confidence': float(face_probs[i])
        })
    
    # Process emotion for each face and create visualization
    for i, frame in enumerate(frames):
        frame_copy = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for face in frame_faces[i]:
            x1, y1, x2, y2 = face['bbox']
            
            # Extract face region for emotion prediction
            face_region = frame_rgb[y1:y2, x1:x2]
            
            # Process face for emotion prediction if region is valid
            if face_region.size > 0:
                face_tensor = preprocess_face(face_region)
                emotion, probs = predict_emotion(model, face_tensor, DEVICE)
                color = EMOTION_COLORS[emotion]
                
                # Draw rectangle and emotion
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
                cv2.putText(frame_copy, emotion_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                # Just draw bounding box if face region isn't valid
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add image dimensions text
        dim_text = f"Size: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(frame_copy, dim_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames_with_emotion.append(frame_copy)
      # Visualize frames with emotion detection
    fig_emotion = visualize_grid(frames_with_emotion, 
                                titles=[f"Frame {i+1} with emotion" for i in range(len(frames_with_emotion))])
    plt.suptitle("Emotion Detection Results", fontsize=16)
    plt.savefig(os.path.join(output_path, "emotion_detection.png"))
    
    print(f"Visualization complete! All images saved to {output_path}/")

def main():    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize batch face detection process')
    parser.add_argument('--video', required=True, help='Path to the input video file')
    parser.add_argument('--face_model', help='Path to the ONNX face detection model file')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to process (max 4)')
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
        num_frames=args.num_frames,
        frame_offset=args.frame_offset,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
