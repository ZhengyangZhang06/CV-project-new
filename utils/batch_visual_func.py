import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.face_detection import load_face_model, predict_faces

from utils.emotion_recognition import EMOTIONS, EMOTION_COLORS, preprocess_face, predict_emotion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_name_from_path_and_size(face_model_path, input_width, input_height):
    """Determine model name based on model path and input size
    
    Args:
        face_model_path: Path to the ONNX model file
        input_width: Model input width
        input_height: Model input height
    
    Returns:
        str: Model name identifier (e.g., "model_320", "model_640", "model_slim_320")
    """
    model_parts = []
    
    # First try to get name components from the model path filename
    if face_model_path:
        filename = os.path.basename(face_model_path).lower()
        
        # Check for model architecture type
        if "rfb" in filename:
            model_parts.append("rfb")
        elif "slim" in filename:
            model_parts.append("slim")
        else:
            model_parts.append("base")
        
        # Check for input size in filename
        if "320" in filename:
            model_parts.append("320")
        elif "640" in filename:
            model_parts.append("640")
    
    # If no size found in filename, use input dimensions
    if not any(size in model_parts for size in ["320", "640"]):
        if input_width == 320:
            model_parts.append("320")
        elif input_width == 640:
            model_parts.append("640")
        else:
            model_parts.append(f"{input_width}x{input_height}")
    
    # If no architecture type was determined, add one based on dimensions
    if not any(arch in model_parts for arch in ["rfb", "slim", "base"]):
        model_parts.insert(0, "model")
    
    # Create final model name
    if len(model_parts) == 1:
        return f"model_{model_parts[0]}"
    else:
        return "_".join(model_parts)



def extract_video_frames(video_path, num_frames, frame_offset=0, frame_interval=1):
    """Extract frames from video
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        frame_offset: Number of frames to skip from start
        frame_interval: Interval between frames (1=consecutive, 2=every other frame, etc.)
    
    Returns:
        tuple: (frames, video_properties) where video_properties is dict with fps, width, height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return None, None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_props = {
        'fps': fps,
        'width': frame_width,
        'height': frame_height,
        'total_frames': total_frames
    }
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} total frames")
    print(f"Extracting {num_frames} frames starting from frame {frame_offset} (interval: {frame_interval})")
    
    # Check if we have enough frames
    frames_needed = frame_offset + (num_frames - 1) * frame_interval + 1
    if frames_needed > total_frames:
        print(f"Warning: Not enough frames in video. Need {frames_needed}, have {total_frames}")
    
    # Skip frames if needed
    for i in range(frame_offset):
        ret = cap.read()[0]
        if not ret:
            print(f"Error: Could not skip frame {i} (video too short)")
            cap.release()
            return None, None
    
    # Read frames with specified interval
    frames = []
    frames_read = 0
    
    for i in range(num_frames):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_offset + i * frame_interval}")
            break
        
        frames.append(frame)
        frames_read += 1
        
        # Skip frames for interval (except for the last frame)
        if i < num_frames - 1:
            for _ in range(frame_interval - 1):
                ret = cap.read()[0]
                if not ret:
                    print(f"Warning: Reached end of video while skipping interval frames")
                    break
    
    cap.release()
    
    print(f"Successfully extracted {frames_read} frames")
    
    if len(frames) < num_frames:
        print(f"Warning: Could only read {len(frames)} frames out of {num_frames} requested")
        # Pad with duplicate frames if needed
        while len(frames) < num_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
        print(f"Padded to {len(frames)} frames using frame duplication")
    
    return frames, video_props

def process_face_detection(grid_image, ort_session, input_name, input_width, input_height, prob_threshold=0.6):
    """Process face detection on grid image
    
    Args:
        grid_image: Input grid image
        ort_session: ONNX runtime session
        input_name: Input tensor name
        input_width: Model input width
        input_height: Model input height
        prob_threshold: Confidence threshold for face detection
    
    Returns:
        tuple: (face_boxes, face_probs, processed_grid)
    """
    # Preprocess the grid for inference
    grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    processed_grid = cv2.resize(grid_rgb, (input_width, input_height))
    
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
    grid_h, grid_w = grid_image.shape[:2]
    face_boxes, _, face_probs = predict_faces(grid_w, grid_h, confidences, boxes, prob_threshold=prob_threshold)
    
    return face_boxes, face_probs, processed_grid

def process_emotions_for_frames(frames, frame_faces, model):
    """Process emotion detection for all faces in frames
    
    Args:
        frames: List of original frames
        frame_faces: List of face data for each frame
        model: Emotion recognition model
    
    Returns:
        list: Frames with emotion annotations
    """
    frames_with_emotion = []
    
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add confidence text
                conf_text = f"Conf: {face['confidence']:.2f}"
                cv2.putText(frame_copy, conf_text, (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame information
        frame_info = f"Frame {i+1}/{len(frames)}"
        cv2.putText(frame_copy, frame_info, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add image dimensions text
        dim_text = f"Size: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(frame_copy, dim_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add face count
        faces_text = f"Faces: {len(frame_faces[i])}"
        cv2.putText(frame_copy, faces_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames_with_emotion.append(frame_copy)
    
    return frames_with_emotion
