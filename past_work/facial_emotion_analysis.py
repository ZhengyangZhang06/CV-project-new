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
def process_frame_with_onnx(frame, face_detector, ort_session, input_name, predictor, emotion_model, device, previous_face_regions):
    """Process a frame using ONNX face detection and emotion prediction"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Preprocess image for ONNX face detection
    input_width = 320
    input_height = 240  # Changed from 320 to 240
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
        
        # Extract face region for landmark detection
        # Convert ONNX detection to dlib rectangle format for landmark prediction
        face_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        
        # Get facial landmarks using dlib predictor
        landmarks = predictor(gray, face_rect)
        
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
            'landmarks': landmarks,
            'confidence': float(face_probs[i])
        })
        
        # Draw rectangle and emotion on the frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add emotion text with percentage
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw facial landmarks
        for i in range(68):
            landmark_x = landmarks.part(i).x
            landmark_y = landmarks.part(i).y
            cv2.circle(frame, (landmark_x, landmark_y), 1, (0, 255, 0), -1)
    
    return frame, results

# Process video with ONNX face detection and emotion recognition
def process_video_with_onnx(video_path, emotion_model, device, output_path=None, sample_rate=15, display=True, draw_landmarks=False):
    """Process video file for emotion detection using ONNX face detection"""
    # Load ONNX face detection model
    onnx_path = None
    
    # Try to find the ONNX model in common locations
    model_dirs = [
        ".",
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
        print("Please specify the correct path to the ONNX model")
        return
    
    print(f"Using ONNX face detection model: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    # Load dlib landmark predictor (still needed for landmark detection)
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
    
    # Create a progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Decide whether to process this frame
            if frame_count % frame_step == 0:
                # Process the frame with ONNX face detection and emotion prediction
                processed_frame, results = process_frame_with_onnx(
                    frame.copy(), None, ort_session, input_name, predictor, emotion_model, device, previous_face_regions
                )
                processed_count += 1
                previous_results = results
                
                # Update previous_face_regions for the next frame
                previous_face_regions = [(r['bbox'][0], r['bbox'][1], 
                                         r['bbox'][0] + r['bbox'][2], 
                                         r['bbox'][1] + r['bbox'][3]) 
                                        for r in results]
                
                # Update progress bar
                pbar.set_postfix({"Detected faces": len(results)})
                
                # Display if requested
                if display:
                    cv2.imshow('Facial Emotion Analysis', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                
                # Write to output video
                if out:
                    out.write(processed_frame)
            else:
                # For unprocessed frames, use previous results
                annotated_frame = frame.copy()
                
                for result in previous_results:
                    x, y, w, h = result['bbox']
                    emotion = result['emotion']
                    probs = result['probabilities']
                    color = EMOTION_COLORS[emotion]
                    
                    # Draw rectangle and emotion
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
                    cv2.putText(annotated_frame, emotion_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Optionally draw landmarks if available and requested
                    if draw_landmarks and 'landmarks' in result:
                        landmarks = result['landmarks']
                        for i in range(68):
                            landmark_x = landmarks.part(i).x
                            landmark_y = landmarks.part(i).y
                            cv2.circle(annotated_frame, (landmark_x, landmark_y), 1, (0, 255, 0), -1)
                
                # Display if requested
                if display:
                    cv2.imshow('Facial Emotion Analysis', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write to output
                if out:
                    out.write(annotated_frame)
            
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
    parser = argparse.ArgumentParser(description='Facial Emotion Analysis on Video using ONNX face detection')
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
    
    # Process the video with ONNX face detection
    process_video_with_onnx(
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
