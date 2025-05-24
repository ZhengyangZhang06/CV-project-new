import cv2
import numpy as np
from tqdm import tqdm
import os
import subprocess

from utils.emotion_recognition import preprocess_face, predict_emotion, EMOTIONS, EMOTION_COLORS
from utils.face_detection import predict_faces

def process_frame_with_onnx(frame, ort_session, input_name, emotion_model, device, input_width=640, input_height=480):
    """Process a single frame using ONNX face detection and emotion prediction"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = []
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Preprocess image for ONNX face detection
    image = cv2.resize(frame_rgb, (input_width, input_height))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    
    # Run ONNX inference
    confidences, boxes = ort_session.run(None, {input_name: image})
    
    # Process the output to get face bounding boxes
    face_boxes, _, face_probs = predict_faces(width, height, confidences, boxes, prob_threshold=0.6)
    
    # Process each detected face
    for i in range(len(face_boxes)):
        x1, y1, x2, y2 = face_boxes[i]
        w, h = x2 - x1, y2 - y1
        
        # Extract the face region for emotion prediction
        face_region = frame_rgb[y1:y2, x1:x2]
        
        # Skip if face region is empty
        if face_region.size == 0:
            continue
        
        # Process the face for emotion prediction
        face_tensor = preprocess_face(face_region)
        emotion, probs = predict_emotion(emotion_model, face_tensor, device)
        
        # Store results
        results.append({
            'bbox': (x1, y1, w, h),
            'emotion': emotion,
            'probabilities': probs,
            'confidence': float(face_probs[i])
        })
    
    return results

def process_batch_with_onnx(frames, ort_session, input_name, emotion_model, device, input_width=640, input_height=480):
    """Process a batch of frames using grid-based ONNX face detection"""
    if not frames:
        return []
    
    frame_height, frame_width = frames[0].shape[:2]
    batch_size = len(frames)      # Determine grid size based on batch size
    if batch_size == 1:
        grid_rows, grid_cols = 1, 1
    elif batch_size <= 4:
        grid_rows, grid_cols = 2, 2
    elif batch_size <= 9:
        grid_rows, grid_cols = 3, 3
    elif batch_size <= 16:
        grid_rows, grid_cols = 4, 4
    elif batch_size <= 25:
        grid_rows, grid_cols = 5, 5
    elif batch_size <= 36:
        grid_rows, grid_cols = 6, 6
    else:
        # For larger batches, use square grid
        grid_rows = int(np.ceil(np.sqrt(batch_size)))
        grid_cols = grid_rows
    
    # Create grid image
    grid_h = grid_rows * frame_height
    grid_w = grid_cols * frame_width
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill grid with frames
    for i, frame in enumerate(frames):
        if i >= grid_rows * grid_cols:
            break
        row, col = divmod(i, grid_cols)
        y_offset = row * frame_height
        x_offset = col * frame_width
        grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
    
    # Preprocess grid for ONNX
    grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    processed_grid = cv2.resize(grid_rgb, (input_width, input_height))
    image_mean = np.array([127, 127, 127])
    processed_grid = (processed_grid - image_mean) / 128
    processed_grid = np.transpose(processed_grid, [2, 0, 1])
    processed_grid = np.expand_dims(processed_grid, axis=0)
    processed_grid = processed_grid.astype(np.float32)
    
    # Run ONNX inference
    confidences, boxes = ort_session.run(None, {input_name: processed_grid})
    
    # Get face boxes
    face_boxes, _, face_probs = predict_faces(grid_w, grid_h, confidences, boxes, prob_threshold=0.6)
    
    # Map faces back to original frames
    batch_results = [[] for _ in range(batch_size)]
    
    for i in range(len(face_boxes)):
        x1, y1, x2, y2 = face_boxes[i]
        
        # Determine which frame this face belongs to
        frame_row = int(y1 // frame_height)
        frame_col = int(x1 // frame_width)
        batch_idx = frame_row * grid_cols + frame_col
        
        if batch_idx >= batch_size:
            continue
        
        # Adjust coordinates to original frame
        x1_local = x1 - (frame_col * frame_width)
        y1_local = y1 - (frame_row * frame_height)
        x2_local = x2 - (frame_col * frame_width)
        y2_local = y2 - (frame_row * frame_height)
        
        # Clamp to frame bounds
        x1_local = max(0, min(x1_local, frame_width))
        y1_local = max(0, min(y1_local, frame_height))
        x2_local = max(0, min(x2_local, frame_width))
        y2_local = max(0, min(y2_local, frame_height))
        
        w_local = x2_local - x1_local
        h_local = y2_local - y1_local
        
        if w_local <= 0 or h_local <= 0:
            continue
        
        # Extract face region and predict emotion
        original_frame = frames[batch_idx]
        face_region = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)[y1_local:y2_local, x1_local:x2_local]
        
        if face_region.size == 0:
            continue
        
        face_tensor = preprocess_face(face_region)
        emotion, probs = predict_emotion(emotion_model, face_tensor, device)
        
        batch_results[batch_idx].append({
            'bbox': (x1_local, y1_local, w_local, h_local),
            'emotion': emotion,
            'probabilities': probs,
            'confidence': float(face_probs[i])
        })
    
    return batch_results

def apply_face_results_to_frame(frame, face_results):
    """Apply face detection and emotion results to a frame"""
    for face_data in face_results:
        x, y, w, h = face_data['bbox']
        emotion = face_data['emotion']
        probs = face_data['probabilities']
        color = EMOTION_COLORS[emotion]
        
        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def process_video_with_onnx_improved(video_path, emotion_model, device, output_path=None, sample_step=2, 
                                   display=True, face_model_path=None, process_all=False, batch_size=4, debug=False):
    """Improved video processing with batch-then-apply logic
    
    Process frames in batches, then apply results to intermediate frames.
    
    Args:
        video_path: Path to input video file
        emotion_model: Emotion recognition model
        device: Device to run emotion model on (CPU/CUDA)
        output_path: Path to save output video
        sample_step: Step size for frame sampling (1-4): process every N frames
        display: Whether to display processing in window
        face_model_path: Path to ONNX face detection model
        process_all: If True, process all frames regardless of sample_step
        batch_size: Number of frames to process in each batch (1, 4, 9, or 16)
        debug: If True, print debug information about detection results
    """
    from utils.face_detection import load_face_model
      # Validate batch_size
    supported_batch_sizes = [1, 4, 9, 16, 25, 36]
    if batch_size not in supported_batch_sizes:
        print(f"Error: batch_size must be one of {supported_batch_sizes}, got {batch_size}")
        print(f"Automatically adjusting to nearest supported size...")
        # Find the closest supported batch size
        batch_size = min(supported_batch_sizes, key=lambda x: abs(x - batch_size))
        print(f"Using batch_size: {batch_size}")
    
    # Load ONNX face detection model
    result = load_face_model(face_model_path)
    if not result:
        return
    
    ort_session, input_width, input_height = result
    input_name = ort_session.get_inputs()[0].name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      # Calculate sampling
    if process_all:
        frame_step = 1
        print(f"Processing ALL frames (process_all=True)")
    else:
        # Validate sample_step is within allowed range
        sample_step = max(1, min(4, sample_step))
        frame_step = sample_step
    print(f"Video properties: {frame_width}x{frame_height}, {fps:.1f} FPS, {total_frames} total frames")
    print(f"Sampling: processing every {frame_step} frame(s) (sample step: {sample_step})")
    print(f"Batch processing: {batch_size} frames per batch")
    print_batch_size_info(batch_size)    # Setup video writer
    out = None
    if output_path:
        # Keep original FPS to maintain video duration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output video FPS: {fps:.2f} (maintaining original duration)")
    
    # Statistics tracking
    processed_count = 0
    frames_with_faces = 0
    frames_without_faces = 0
    
    # Read all frames first for batch processing
    print("Reading all frames...")
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    total_frames = len(all_frames)
    print(f"Total frames read: {total_frames}")
    
    # Determine frames to process based on sampling
    if process_all:
        frames_to_process_indices = list(range(total_frames))
    else:
        frames_to_process_indices = list(range(0, total_frames, frame_step))
    
    print(f"Frames to process: {len(frames_to_process_indices)} ({len(frames_to_process_indices)/total_frames*100:.1f}% of total)")
      # Store results for processed frames
    frame_results = {}
    
    # Calculate total batches
    total_batches = (len(frames_to_process_indices) + batch_size - 1) // batch_size    # Initialize progress bars
    batch_pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch", position=0)
    frame_pbar = tqdm(total=total_frames, desc="Applying to frames", unit="frame", position=1)
    
    # Track which frames have been written to avoid duplicates
    written_frames = set()
    
    try:
        # Process frames in batches
        batch_start_idx = 0
        current_batch_num = 0
        while batch_start_idx < len(frames_to_process_indices):
            # Get current batch of frame indices
            batch_end_idx = min(batch_start_idx + batch_size, len(frames_to_process_indices))
            current_batch_indices = frames_to_process_indices[batch_start_idx:batch_end_idx]
            current_batch_num += 1
            
            # Update batch progress bar description
            batch_pbar.set_description(f"Processing batch {current_batch_num}/{total_batches} (frames {current_batch_indices})")

            
            # Collect frames for this batch
            batch_frames = []
            for frame_idx in current_batch_indices:
                batch_frames.append(all_frames[frame_idx])
            
            # Process the batch
            batch_results = process_batch_with_onnx(
                batch_frames, ort_session, input_name, 
                emotion_model, device, input_width, input_height
            )
            
            # Store results for each frame in the batch
            for i, results in enumerate(batch_results):
                frame_idx = current_batch_indices[i]
                frame_results[frame_idx] = results
                processed_count += 1
                
                # Track detection statistics
                if results:
                    frames_with_faces += 1
                    if debug:
                        print(f"Frame {frame_idx}: {len(results)} face(s) detected")
                else:
                    frames_without_faces += 1
                    if debug:
                        print(f"Frame {frame_idx}: No faces detected")
              # Update batch progress
            batch_pbar.update(1)
            
            # Apply results to all frames covered by this batch
            # Determine the range of frames this batch covers
            if batch_start_idx == 0:
                # First batch starts from frame 0
                start_frame = 0
            else:
                # Start from the frame after the previous batch's last processed frame
                prev_last_processed = frames_to_process_indices[batch_start_idx - 1]
                start_frame = prev_last_processed + 1
            
            if batch_end_idx >= len(frames_to_process_indices):
                # Last batch covers until the end
                end_frame = total_frames
            else:
                # End at the next batch's first processed frame
                next_first_processed = frames_to_process_indices[batch_end_idx]
                end_frame = next_first_processed
                
            if debug:
                print(f"Applying results to frames {start_frame} to {end_frame-1}")
            
            # Update frame progress bar description
            frame_pbar.set_description(f"Applying batch {current_batch_num} results to frames {start_frame}-{end_frame-1}")
            
            # Apply results to frames in this range
            for frame_idx in range(start_frame, end_frame):
                # Skip if frame already written to avoid duplicates
                if frame_idx in written_frames:
                    continue
                
                frame_copy = all_frames[frame_idx].copy()
                
                # Find the appropriate results to apply
                results_to_apply = []
                result_source = "none"
                
                if frame_idx in frame_results:
                    # This frame was directly processed
                    results_to_apply = frame_results[frame_idx]
                    result_source = "direct"
                else:
                    # Find the most recent processed frame before this one
                    best_frame = -1
                    # First check in current batch
                    for processed_frame_idx in reversed(current_batch_indices):
                        if processed_frame_idx <= frame_idx:
                            best_frame = processed_frame_idx
                            break
                    
                    # If no frame found in current batch, look in all previous results
                    if best_frame == -1:
                        for processed_frame_idx in sorted(frame_results.keys(), reverse=True):
                            if processed_frame_idx < frame_idx:
                                best_frame = processed_frame_idx
                                break
                    
                    if best_frame >= 0:
                        results_to_apply = frame_results[best_frame]
                        result_source = f"inherited_from_{best_frame}"
                        if debug and frame_idx % 50 == 0:
                            print(f"Frame {frame_idx}: Inheriting from frame {best_frame}")
                
                # Apply the results
                if results_to_apply:
                    apply_face_results_to_frame(frame_copy, results_to_apply)
                    if debug and frame_idx % 50 == 0:
                        print(f"Frame {frame_idx}: Applied {len(results_to_apply)} detection(s) ({result_source})")
                
                # Display
                if display:
                    cv2.imshow('Facial Emotion Analysis', frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                
                # Write to output
                if out:
                    out.write(frame_copy)
                    written_frames.add(frame_idx)
                
                # Update frame progress
                frame_pbar.update(1)
              # Move to next batch
            batch_start_idx = batch_end_idx
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    
    finally:
        # Close progress bars
        batch_pbar.close()
        frame_pbar.close()
          # Print statistics
        processing_ratio = processed_count / total_frames * 100 if total_frames > 0 else 0
        expected_ratio = len(frames_to_process_indices) / total_frames * 100 if total_frames > 0 else 0
        
        print("\n--- Processing Statistics ---")
        print(f"Total frames: {total_frames}")
        print(f"Processed frames: {processed_count} ({processing_ratio:.1f}%)")
        print(f"Output frames: {len(written_frames)} (all frames with inherited results)")
        print(f"Frames with faces: {frames_with_faces}")
        print(f"Frames without faces: {frames_without_faces}")        
        print(f"Face detection rate: {frames_with_faces/processed_count*100:.1f}% of processed frames" if processed_count > 0 else "N/A")
        print(f"Expected processing rate: {expected_ratio:.1f}%")        
        print(f"Sampling step: {sample_step} (video: {fps:.1f} FPS)")
        if output_path:
            print(f"Output video FPS: {fps:.1f} (maintained original duration)")
        print(f"Batch processing: {batch_size} frames per batch")
        print_batch_size_info(batch_size)
        print(f"Total batches processed: {current_batch_num}/{total_batches}")
        
        accuracy_diff = abs(processing_ratio - expected_ratio)
        if accuracy_diff < 2:
            print("✓ Batch processing working correctly!")
        else:
            print(f"⚠ Processing ratio differs by {accuracy_diff:.1f}% from expected")
        
        # Clear results to free memory
        frame_results.clear()
        
        if out:
            out.release()
        cv2.destroyAllWindows()

# Legacy function for backward compatibility
def process_video_with_onnx(video_path, emotion_model, device, output_path=None, sample_step=2, 
                           display=True, face_model_path=None, process_all=False, batch_size=4):
    """Legacy function - calls improved version"""
    return process_video_with_onnx_improved(
        video_path, emotion_model, device, output_path, sample_step, 
        display, face_model_path, process_all, batch_size
    )

def print_batch_size_info(batch_size):
    """Print information about the grid layout for the given batch size"""
    grid_layouts = {
        1: (1, 1, "Single frame processing"),
        4: (2, 2, "2x2 grid - optimal for most cases"),
        9: (3, 3, "3x3 grid - good for larger batches"),
        16: (4, 4, "4x4 grid - maximum efficiency for large batches"),
        25: (5, 5, "5x5 grid - high throughput processing"),
        36: (6, 6, "6x6 grid - maximum batch efficiency")
    }
    
    if batch_size in grid_layouts:
        rows, cols, description = grid_layouts[batch_size]
        print(f"Batch size {batch_size}: {rows}x{cols} grid layout - {description}")
    else:
        print(f"Batch size {batch_size}: Custom grid layout")
