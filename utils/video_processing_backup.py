import cv2
import numpy as np
from tqdm import tqdm
import os
import subprocess

from utils.emotion_recognition import preprocess_face, predict_emotion, EMOTIONS, EMOTION_COLORS
from utils.face_detection import predict_faces

def process_frame_with_onnx(frame, ort_session, input_name, emotion_model, device, previous_face_regions, input_width=640, input_height=480):
    """Process a frame using ONNX face detection and emotion prediction"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Preprocess image for ONNX face detection using the provided dimensions
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
        
        # Draw rectangle and emotion on the frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add emotion text with percentage
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame, results

def process_video_with_onnx(video_path, emotion_model, device, output_path=None, sample_rate=15, 
                           display=True, face_model_path=None, process_all=False):
    """Process video file for emotion detection using ONNX face detection with batch processing
    
    Args:
        video_path: Path to input video file
        emotion_model: Emotion recognition model
        device: Device to run emotion model on (CPU/CUDA)
        output_path: Path to save output video
        sample_rate: Target frames per second to process
        display: Whether to display processing in window
        face_model_path: Path to ONNX face detection model
        process_all: If True, process all frames regardless of sample_rate
    """
    from utils.face_detection import load_face_model
    
    # Load ONNX face detection model
    result = load_face_model(face_model_path)
    if not result:
        return
    
    # Unpack the model session and input dimensions
    ort_session, input_width, input_height = result
        
    input_name = ort_session.get_inputs()[0].name
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      # Calculate frame sampling
    if process_all:
        frame_step = 1  # Process every frame
        print(f"Processing ALL frames (process_all=True)")
    else:
        frame_step = max(1, int(fps / sample_rate))
        
    # Add more detailed debugging output
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} total frames")
    print(f"Sampling rate: {sample_rate} samples per second (1 frame processed every {frame_step} frames)")
    expected_processed = total_frames // frame_step
    print(f"Expected frames to process: ~{expected_processed} (approximately {expected_processed/total_frames*100:.1f}% of total frames)")
    print(f"Batch processing: reading 8 frames and processing up to 4 frames at once")
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processed_count = 0
    previous_results = {}  # 存储每个处理帧的结果，键为绝对帧位置
    last_processed_frame = None  # 用于跟踪最后处理的帧的索引
      # Create a progress bar
    pbar = tqdm(total=total_frames, desc=f"Processing video (1/{frame_step} frames)", unit="frames")
    
    try:
        while cap.isOpened():
            batch_frames = []  # Frames that need processing
            batch_indices = []  # Indices of frames that need processing
            all_original_frames = []  # All original frames
            valid_frames = 0
              # Read 8 frames, but select only those that need processing based on the sampling rate
            for i in range(8):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                valid_frames += 1
                all_original_frames.append(frame.copy())
                
                # Mark frames that need processing based on sampling rate
                if frame_count % frame_step == 0:
                    batch_frames.append(frame.copy())
                    batch_indices.append(valid_frames - 1)  # Store index in all_original_frames
                    # 跟踪最后处理的帧
                    last_processed_frame = frame_count
                
                pbar.update(1)
            
            if valid_frames == 0:
                break  # End of video
            
            # Process the batch of frames if we have any frames to process
            if batch_frames:
                # Limit to max 4 frames for batch processing, but if we have more,
                # process them in multiple batches to ensure all requested frames are processed
                while batch_frames:
                    # Get up to 4 frames for this batch iteration
                    current_batch_frames = batch_frames[:4]
                    current_batch_indices = batch_indices[:4]
                    
                    # Remove the frames we're about to process from the pending list
                    batch_frames = batch_frames[4:] if len(batch_frames) > 4 else []
                    batch_indices = batch_indices[4:] if len(batch_indices) > 4 else []
                    
                    # Create a 2x2 grid of frames for batch processing
                    grid_h = 2 * frame_height
                    grid_w = 2 * frame_width
                    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                  # Fill the grid with frames (all should be valid)
                for i, frame in enumerate(current_batch_frames):
                    row, col = divmod(i, 2)
                    y_offset = row * frame_height
                    x_offset = col * frame_width
                    grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
                  # Preprocess the grid image for ONNX face detection using the model's expected dimensions
                grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
                processed_grid = cv2.resize(grid_rgb, (input_width, input_height))
                image_mean = np.array([127, 127, 127])
                processed_grid = (processed_grid - image_mean) / 128
                processed_grid = np.transpose(processed_grid, [2, 0, 1])
                processed_grid = np.expand_dims(processed_grid, axis=0)
                processed_grid = processed_grid.astype(np.float32)
                
                # Run ONNX inference on the grid
                confidences, boxes = ort_session.run(None, {input_name: processed_grid})
                
                # Process the output to get face bounding boxes
                face_boxes, _, face_probs = predict_faces(grid_w, grid_h, confidences, boxes, prob_threshold=0.6)
                  # Process each detected face and map back to original frames
                batch_results = [{} for _ in range(len(current_batch_frames))]
                
                for i in range(len(face_boxes)):
                    x1, y1, x2, y2 = face_boxes[i]
                    
                    # Determine which frame this face belongs to
                    frame_row = int(y1 // frame_height)
                    frame_col = int(x1 // frame_width)
                    batch_idx = frame_row * 2 + frame_col
                    
                    if batch_idx >= len(current_batch_frames):
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
                    
                    # Extract the face region for emotion prediction
                    original_frame = current_batch_frames[batch_idx]
                    face_region = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)[y1_local:y2_local, x1_local:x2_local]
                    
                    # Skip if face region is empty
                    if face_region.size == 0:
                        continue
                    
                    # Process the face for emotion prediction
                    face_tensor = preprocess_face(face_region)
                    emotion, probs = predict_emotion(emotion_model, face_tensor, device)
                    
                    # Store results for this frame
                    if 'faces' not in batch_results[batch_idx]:
                        batch_results[batch_idx]['faces'] = []
                    
                    batch_results[batch_idx]['faces'].append({
                        'bbox': (x1_local, y1_local, w_local, h_local),
                        'emotion': emotion,
                        'probabilities': probs,
                        'confidence': float(face_probs[i])
                    })
                  # Now apply results to the original processed frames and store for later use
                for i, batch_idx in enumerate(current_batch_indices):
                    if i >= len(batch_results):
                        break
                        
                    processed_count += 1
                    original_frame = all_original_frames[batch_idx]
                    frame_result = batch_results[i]
                    
                    # Store results for future use by unprocessed frames
                    # 使用绝对帧位置作为键
                    frame_position = frame_count - valid_frames + batch_idx + 1  # Calculate absolute frame position
                    previous_results[frame_position] = frame_result
                    
                    # Apply results to frame
                    if 'faces' in frame_result:
                        for face_data in frame_result['faces']:
                            x, y, w, h = face_data['bbox']
                            emotion = face_data['emotion']
                            probs = face_data['probabilities']
                            color = EMOTION_COLORS[emotion]
                            
                            # Draw rectangle and emotion
                            cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, 2)
                            emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
                            cv2.putText(original_frame, emotion_text, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Now process all frames in the batch (including unprocessed ones)
                for i in range(valid_frames):
                    original_frame = all_original_frames[i]
                    frame_position = frame_count - valid_frames + i + 1  # Calculate absolute frame position
                    
                    # For frames that weren't processed, find nearest processed frame
                    if frame_position % frame_step != 0:
                        # 寻找最近的已处理帧结果
                        # 计算应该使用的参考帧 - 向前取最近的处理帧
                        reference_frame = (frame_position // frame_step) * frame_step
                        if reference_frame == 0:  # 如果是视频最开始的帧，使用之后的第一个处理帧
                            reference_frame = frame_step
                            
                        # 获取参考帧的结果
                        prev_result = previous_results.get(reference_frame, {})
                        
                        if 'faces' in prev_result:                            
                            for face_data in prev_result['faces']:
                                x, y, w, h = face_data['bbox']
                                emotion = face_data['emotion']
                                probs = face_data['probabilities']
                                color = EMOTION_COLORS[emotion]
                                
                                # Draw rectangle and emotion
                                cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, 2)
                                emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
                                cv2.putText(original_frame, emotion_text, (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Display if requested
                    if display:
                        cv2.imshow('Facial Emotion Analysis', original_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            raise KeyboardInterrupt
                    
                    # Write to output video
                    if out:
                        out.write(original_frame)
            
            else:
                # No frames need processing in this batch, use previous results for all frames
                for i in range(valid_frames):
                    original_frame = all_original_frames[i]
                    frame_position = frame_count - valid_frames + i + 1
                    
                    # 寻找最近的已处理帧结果
                    # 计算应该使用的参考帧 - 向前取最近的处理帧
                    reference_frame = (frame_position // frame_step) * frame_step
                    if reference_frame == 0:  # 如果是视频最开始的帧，使用之后的第一个处理帧
                        reference_frame = frame_step
                        
                    # 获取参考帧的结果
                    prev_result = previous_results.get(reference_frame, {})
                    
                    if 'faces' in prev_result:
                        for face_data in prev_result['faces']:
                            x, y, w, h = face_data['bbox']
                            emotion = face_data['emotion']
                            probs = face_data['probabilities']
                            color = EMOTION_COLORS[emotion]
                            
                            # Draw rectangle and emotion
                            cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, 2)
                            emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
                            cv2.putText(original_frame, emotion_text, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Display if requested
                    if display:
                        cv2.imshow('Facial Emotion Analysis', original_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            raise KeyboardInterrupt
                    
                    # Write to output video
                    if out:
                        out.write(original_frame)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    
    finally:
        pbar.close()
        # 添加更详细的统计信息
        processing_ratio = processed_count / frame_count * 100 if frame_count > 0 else 0
        expected_ratio = 100 / frame_step if frame_step > 0 else 100
        
        print("\n--- Processing Statistics ---")
        print(f"Total frames in video: {frame_count}")
        print(f"Processed frames: {processed_count} ({processing_ratio:.1f}%)")
        print(f"Expected processing rate: 1 frame every {frame_step} frames ({expected_ratio:.1f}%)")
        print(f"Sampling rate: {sample_rate} frames per second")
        print(f"Video FPS: {fps}")
        
        if abs(processing_ratio - expected_ratio) > 5:  # 5% 误差容忍
            print("\nNOTE: The actual processing ratio differs from the expected ratio.")
            print("This could be due to batch processing limitations or end-of-video handling.")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

