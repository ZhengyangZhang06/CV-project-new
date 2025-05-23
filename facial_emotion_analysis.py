import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import onnxruntime as ort

# Constants from detect_emotions.py
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels and colors
EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
EMOTION_COLORS = {
    0: (255, 0, 0), 1: (128, 0, 128), 2: (128, 0, 255), 3: (0, 255, 0),
    4: (0, 0, 255), 5: (255, 255, 0), 6: (192, 192, 192)
}

# ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Functions for emotion prediction
def preprocess_face(face_img):
    # Convert to grayscale and resize
    face_pil = Image.fromarray(face_img).convert('L')
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    return transform(face_pil).unsqueeze(0)

def predict_emotion(model, face_tensor, device):
    model.eval()
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probs.cpu().numpy()[0]

# ONNX Face detection utilities
def predict_faces(width, height, confidences, boxes, prob_threshold=0.7, iou_threshold=0.3, top_k=-1):
    """Process raw ONNX model output to get face bounding boxes"""
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        
        if probs.shape[0] == 0:
            continue
            
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        
        # Apply non-maximum suppression
        keep = nms(box_probs, iou_threshold)
        if keep.shape[0] > 0:
            box_probs = box_probs[keep, :]
            
        if top_k > 0:
            if box_probs.shape[0] > top_k:
                box_probs = box_probs[:top_k, :]
                
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
        
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
        
    picked_box_probs = np.concatenate(picked_box_probs)
    
    # Convert normalized coordinates to image dimensions
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def nms(box_probs, iou_threshold=0.3, top_k=-1):
    """Non-maximum suppression"""
    keep = []
    
    # Sort boxes by confidence
    order = np.argsort(box_probs[:, 4])[::-1]
    
    # Apply NMS
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if top_k > 0 and len(keep) >= top_k:
            break
            
        xx1 = np.maximum(box_probs[i, 0], box_probs[order[1:], 0])
        yy1 = np.maximum(box_probs[i, 1], box_probs[order[1:], 1])
        xx2 = np.minimum(box_probs[i, 2], box_probs[order[1:], 2])
        yy2 = np.minimum(box_probs[i, 3], box_probs[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = (box_probs[i, 2] - box_probs[i, 0]) * (box_probs[i, 3] - box_probs[i, 1]) + \
                (box_probs[order[1:], 2] - box_probs[order[1:], 0]) * \
                (box_probs[order[1:], 3] - box_probs[order[1:], 1]) - intersection
        
        iou = intersection / np.maximum(union, 1e-10)
        mask = iou <= iou_threshold
        
        order = order[1:][mask]
        
    return np.array(keep)

# Function to detect faces using ONNX model and predict emotions
def process_frame_with_onnx(frame, ort_session, input_name, emotion_model, device, previous_face_regions):
    """Process a frame using ONNX face detection and emotion prediction"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Preprocess image for ONNX face detection
    input_width = 640
    input_height = 480  # Changed from 320 to 240
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

# Process video with ONNX face detection and emotion recognition using batch processing
def process_video_with_onnx(video_path, emotion_model, device, output_path=None, sample_rate=15, 
                           display=True, face_model_path=None):
    """Process video file for emotion detection using ONNX face detection with batch processing"""
    # Load ONNX face detection model
    onnx_path = None
    
    # Use the provided model path if available
    if face_model_path and os.path.exists(face_model_path):
        onnx_path = face_model_path
        print(f"Using provided model path: {onnx_path}")
    else:
        # Try to find the ONNX model in common locations
        model_dirs = [
            ".",
            "./models",
            "./models/onnx",
            "Ultra-Light-Fast-Generic-Face-Detector-1MB/models/onnx"
        ]
        
        for model_dir in model_dirs:
            for model_name in ["version-RFB-320.onnx", "version-slim-320.onnx"]:
                path = os.path.join(model_dir, model_name)
                if os.path.exists(path):
                    onnx_path = path
                    break
            if onnx_path:
                break
    
    if not onnx_path:
        print("Error: Could not find ONNX face detection model")
        print("Please specify the correct path to the ONNX model using --face_model")
        return
    
    print(f"Using ONNX face detection model: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
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
    frame_step = max(1, int(fps / sample_rate))
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS")
    print(f"Processing every {frame_step} frames to achieve {sample_rate} samples per second")
    print(f"Batch processing: 4 frames at once")
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processed_count = 0
    previous_results = [{}, {}, {}, {}]  # Store results for each frame position
    
    # Create a progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    
    try:
        while cap.isOpened():
            batch_frames = []
            batch_frames_original = []
            valid_frames = 0
            
            # Read 4 frames for batch processing
            for _ in range(4):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                valid_frames += 1
                batch_frames_original.append(frame.copy())
                
                # Only process every frame_step frames
                if frame_count % frame_step == 0:
                    batch_frames.append(frame.copy())
                else:
                    # For frames we wouldn't normally process, add a placeholder
                    batch_frames.append(None)
                
                pbar.update(1)
            
            if valid_frames == 0:
                break  # End of video
            
            # Process the batch of frames
            if any(frame is not None for frame in batch_frames):
                # Create a 2x2 grid of frames for batch processing
                grid_h = 2 * frame_height
                grid_w = 2 * frame_width
                grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                
                # Fill the grid with frames (if available)
                for i, frame in enumerate(batch_frames):
                    if frame is not None:
                        row, col = divmod(i, 2)
                        y_offset = row * frame_height
                        x_offset = col * frame_width
                        grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
                
                # Preprocess the grid image for ONNX face detection
                grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
                input_width = 640
                input_height = 480
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
                frame_results = [{}, {}, {}, {}]
                
                for i in range(len(face_boxes)):
                    x1, y1, x2, y2 = face_boxes[i]
                    
                    # Determine which frame this face belongs to
                    frame_row = int(y1 // frame_height)
                    frame_col = int(x1 // frame_width)
                    frame_idx = frame_row * 2 + frame_col
                    
                    if frame_idx >= valid_frames or batch_frames[frame_idx] is None:
                        continue  # Skip faces in invalid frames or frames we're not processing
                    
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
                    original_frame = batch_frames_original[frame_idx]
                    face_region = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)[y1_local:y2_local, x1_local:x2_local]
                    
                    # Skip if face region is empty
                    if face_region.size == 0:
                        continue
                    
                    # Process the face for emotion prediction
                    face_tensor = preprocess_face(face_region)
                    emotion, probs = predict_emotion(emotion_model, face_tensor, device)
                    
                    # Store results for this frame
                    if 'faces' not in frame_results[frame_idx]:
                        frame_results[frame_idx]['faces'] = []
                    
                    frame_results[frame_idx]['faces'].append({
                        'bbox': (x1_local, y1_local, w_local, h_local),
                        'emotion': emotion,
                        'probabilities': probs,
                        'confidence': float(face_probs[i])
                    })
                
                # Now process each original frame with its corresponding results
                for i in range(valid_frames):
                    if batch_frames[i] is None:  # Skip frames we're not processing
                        continue
                    
                    original_frame = batch_frames_original[i]
                    frame_result = frame_results[i]
                    processed_count += 1
                    
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
                    
                    # Store results for this frame for future use
                    if frame_count % frame_step == 0:
                        previous_results[i % 4] = frame_result
                    
                    # Display if requested
                    if display:
                        cv2.imshow('Facial Emotion Analysis', original_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            raise KeyboardInterrupt
                    
                    # Write to output video
                    if out:
                        out.write(original_frame)
            
            else:
                # For frames we're not processing, use the previous results
                for i in range(valid_frames):
                    original_frame = batch_frames_original[i]
                    
                    # Use the appropriate previous result based on frame position
                    prev_result = previous_results[i % 4]
                    
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
        print(f"\nProcessed {processed_count} frames out of {frame_count} total frames")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial Emotion Analysis on Video using ONNX face detection')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video (optional)')
    parser.add_argument('--sample_rate', type=int, default=15, help='Frames per second to process (default: 15)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display during processing')
    parser.add_argument('--face_model', help='Path to the ONNX face detection model file (optional)')
    args = parser.parse_args()
    
    # Load the emotion recognition model
    model_path = 'models/fer2013_resnet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded emotion recognition model from {model_path}")
    
    # Get video path if not provided as argument
    video_path = args.video
    if not video_path:
        video_path = input("Enter the path to your video file: ")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    # Get face model path if not provided as argument
    face_model_path = args.face_model
    if not face_model_path and args.video is None:  # interactive mode
        model_option = input("Do you want to specify a path to the face detection model? (y/n): ").strip().lower()
        if model_option == 'y':
            face_model_path = input("Enter the path to the ONNX face detection model: ").strip()
    
    # Get output path if not provided
    output_path = args.output
    if not output_path:
        save_option = input("Do you want to save the processed video? (y/n): ").strip().lower()
        if save_option == 'y':
            output_path = input("Enter the output path (or press Enter for default): ").strip()
            if not output_path:
                output_path = 'output_' + os.path.basename(video_path)
    
    # Ask if user wants to see the processing (only in interactive mode)
    display = not args.no_display
    if args.video is None:  # If we're in interactive mode
        display_option = input("Do you want to display video while processing? (y/n): ").strip().lower()
        display = display_option == 'y'
    
    # Process the video with ONNX face detection
    process_video_with_onnx(
        video_path, 
        model, 
        DEVICE,
        output_path=output_path,
        sample_rate=args.sample_rate,
        display=display,
        face_model_path=face_model_path
    )
    
    print("Video processing complete!")

if __name__ == "__main__":
    main()
