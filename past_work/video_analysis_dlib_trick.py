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

# Constants from detect_emotions.py
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device('cpu') # "cuda" if torch.cuda.is_available() else "cpu"

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

# Function to detect faces using dlib and predict emotions
def process_frame_with_tracking(frame, detector, predictor, emotion_model, device, previous_face_regions, suggested_scale_factor=None):
    """Process a frame using face tracking based on previous face locations"""
    # Calculate dynamic scale_factor to resize frame to 480p height
    target_height = 480
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    frame_area = frame_height * frame_width
    
    # Use suggested scale factor if provided and in pre, have faces detected, otherwise calculate a new one
    if suggested_scale_factor is not None and previous_face_regions:
        scale_factor = suggested_scale_factor
    else:
        scale_factor = min(1.0, target_height / frame_height)  # Don't upscale small frames
    
    # Apply scale factor if significant
    if scale_factor < 0.8: # only resize if the frame is not too near to 480p
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        scale_factor = 1.0
        small_frame = frame.copy()
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = []
    frame_height, frame_width = gray.shape
    
    # Check if we have previous face regions to use for targeted detection
    if previous_face_regions:
        all_small_faces = []
        
        # Process each previous face region with padding
        for prev_face in previous_face_regions:
            x1, y1, x2, y2 = prev_face
            
						# Calculate dynamic padding based on face size (30% of face dimensions)
            face_width = abs(x2 - x1)
            face_height = abs(y2 - y1)
            pad = max(30, int(1.0 * max(face_width, face_height)))
            x1_padded = max(0, x1 - pad)
            y1_padded = max(0, y1 - pad)
            x2_padded = min(frame_width, x2 + pad)
            y2_padded = min(frame_height, y2 + pad)
            
            # Extract region of interest
            roi = gray[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Skip if ROI is empty
            if roi.size == 0:
                continue
                
            # Detect faces in this region
            faces_in_roi = detector(roi)
            
            # Adjust coordinates back to full frame
            for face in faces_in_roi:
                adjusted_face = dlib.rectangle(
                    face.left() + x1_padded,
                    face.top() + y1_padded,
                    face.right() + x1_padded,
                    face.bottom() + y1_padded
                )
                all_small_faces.append(adjusted_face)
        
        # Don't do full frame detection if no faces found in ROIs since we think when this happens, the face is not in the frame anymore, and new faces are into the frame, in the next detection, we will do full frame detection. So we only loss the first frame of the new faces.
        #if not all_faces:
        #    all_faces = detector(gray)
    else:
        # No previous faces, so we need to do full frame detection
        all_small_faces = detector(gray)
    """
    # directly do full frame detection
    all_small_faces = detector(gray)"""
    all_faces = []
    for face in all_small_faces:
        x1 = int(face.left() / scale_factor)
        y1 = int(face.top() / scale_factor)
        x2 = int(face.right() / scale_factor)
        y2 = int(face.bottom() / scale_factor)
        all_faces.append(dlib.rectangle(x1, y1, x2, y2))
    
    # Process each detected face
    for face in all_faces:
        # Convert dlib rectangle to OpenCV rectangle format
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()
        
        # Get facial landmarks - but we need to do this in the scaled coordinates
        # Create a scaled face for landmarks detection
        scaled_face = dlib.rectangle(
            int(x * scale_factor), 
            int(y * scale_factor),
            int((x+w) * scale_factor),
            int((y+h) * scale_factor)
        )
        landmarks = predictor(gray, scaled_face)
        
        # Extract the face region
        face_region = frame_rgb[y:y+h, x:x+w]
        
        # Skip if face region is empty
        if face_region.size == 0:
            continue
        
        # Process the face for emotion prediction
        face_tensor = preprocess_face(face_region)
        emotion, probs = predict_emotion(emotion_model, face_tensor, device)
        
        # Store results
        results.append({
            'bbox': (x, y, w, h),
            'emotion': emotion,
            'probabilities': probs,
            'landmarks': landmarks,
            'scale_factor': scale_factor
        })
        
        # Draw rectangle and emotion on the frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add emotion text with percentage
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw facial landmarks
        for i in range(68):
            # The landmarks are in the small frame's coordinate space, so we need to convert them back
            landmark_x = int(landmarks.part(i).x / scale_factor)
            landmark_y = int(landmarks.part(i).y / scale_factor)
            cv2.circle(frame, (landmark_x, landmark_y), 1, (0, 255, 0), -1)
    
    # After processing all faces, calculate new scale factor for next frame
    next_scale_factor = scale_factor
    if all_faces:  # Only if we detected faces
        total_face_area = 0
        face_resolution_sufficient = False
        
        # Calculate total face area and check if any face has sufficient resolution
        for face in all_faces:
            w, h = face.width(), face.height()
            face_area = w * h
            total_face_area += face_area
            
            # Check if face has resolution greater than 96x96
            if w > 96 and h > 96:
                face_resolution_sufficient = True
        
        # Calculate ratio of total face area to frame area
        face_to_frame_ratio = total_face_area / frame_area
        
        # If faces occupy significant space (>15%) and have sufficient resolution,
        # increase scale factor to reduce resolution in next frame
        if face_to_frame_ratio > 0.05 and face_resolution_sufficient:
            next_scale_factor = scale_factor * 0.5  # Reduce resolution by 50%
            print(f"Reducing scale factor to {next_scale_factor:.2f} based on face area")
            next_scale_factor = min(0.8, next_scale_factor)  # Don't reduce too much
    
    return frame, results, next_scale_factor

def apply_previous_detections(current_frame, previous_frame, previous_results):
    """Apply previous face detections to the current frame by copying face regions."""
    if not previous_results:
        return current_frame
    
    # Create a copy of the current frame to work on
    result_frame = current_frame.copy()
    
    for result in previous_results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        probs = result['probabilities']
        
        # Define the face region
        face_region = (slice(y, y+h), slice(x, x+w))
        
        # For visualization purposes, we'll still draw the rectangle and emotion text on the new frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(result_frame, emotion_text, (x, y-10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw landmarks if available
        if 'landmarks' in result:
            landmarks = result['landmarks']
            scale_factor = result['scale_factor']
            for i in range(68):
                # We already computed the correct coordinates when storing the landmarks
                landmark_x = int(landmarks.part(i).x / scale_factor)
                landmark_y = int(landmarks.part(i).y / scale_factor)
                cv2.circle(result_frame, (landmark_x, landmark_y), 1, (0, 255, 0), -1)
    
    return result_frame

# Process video with dlib face detection and emotion recognition
def process_video(video_path, emotion_model, device, output_path=None, sample_rate=15, display=True, draw_landmarks=False):
    """Process video file for emotion detection using dlib"""
    # Load face detector and landmark predictor from dlib
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        print(f"Error: Landmark predictor file '{predictor_path}' not found.")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    predictor = dlib.shape_predictor(predictor_path)
    
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
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processed_count = 0
    previous_results = []  # Store the most recent detection results
    previous_face_regions = []  # Store previous face regions
    last_processed_frame = None  # Store the last processed frame
    current_scale_factor = None  # Store the dynamic scale factor
    
    # Create a progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Decide whether to process this frame
            if frame_count % frame_step == 0:
                # Process the frame with dlib face detection and emotion prediction
                processed_frame, results, next_scale_factor = process_frame_with_tracking(
                    frame.copy(), detector, predictor, emotion_model, device, 
                    previous_face_regions, current_scale_factor
                )
                current_scale_factor = next_scale_factor  # Update scale factor for next frame
                processed_count += 1
                previous_results = results
                last_processed_frame = processed_frame.copy()
                
                # Update previous_face_regions for the next frame
                previous_face_regions = [(r['bbox'][0], r['bbox'][1], 
                                         r['bbox'][0] + r['bbox'][2], 
                                         r['bbox'][1] + r['bbox'][3]) 
                                        for r in results]
                
                # Update progress bar
                pbar.set_postfix({"Detected faces": len(results), "Scale": f"{current_scale_factor:.2f}"})
                
                # Display if requested
                if display:
                    cv2.imshow('Facial Emotion Analysis', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                
                # Write to output video
                if out:
                    out.write(processed_frame)
            else:
                # For unprocessed frames, apply the previous results more smoothly
                if last_processed_frame is not None:
                    # Apply the previous detections to the current frame
                    annotated_frame = apply_previous_detections(
                        frame.copy(), last_processed_frame, previous_results
                    )
                    
                    # Display if requested
                    if display:
                        cv2.imshow('Facial Emotion Analysis', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Write to output
                    if out:
                        out.write(annotated_frame)
                else:
                    # If we don't have a processed frame yet, just write the original frame
                    if display:
                        cv2.imshow('Facial Emotion Analysis', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    if out:
                        out.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
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
    parser = argparse.ArgumentParser(description='Facial Emotion Analysis on Video using dlib')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video (optional)')
    parser.add_argument('--sample_rate', type=int, default=15, help='Frames per second to process (default: 15)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display during processing')
    parser.add_argument('--landmarks', action='store_true', help='Draw facial landmarks')
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
        
    print(f"device: {DEVICE}")
    # print the device name
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    # Process the video
    process_video(
        video_path, 
        model, 
        DEVICE,
        output_path=output_path,
        sample_rate=args.sample_rate,
        display=display,
        draw_landmarks=args.landmarks
    )
    
    print("Video processing complete!")

if __name__ == "__main__":
    main()
