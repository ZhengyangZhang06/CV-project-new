import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Constants
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

def detect_and_predict_emotions(image_path, model, device):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)
    results = []
    for (x, y, w, h) in faces:
        face_tensor = preprocess_face(img_rgb[y:y+h, x:x+w])
        emotion, probs = predict_emotion(model, face_tensor, device)
        results.append({'bbox': (x, y, w, h), 'emotion': emotion, 'probabilities': probs})
    return img_rgb, results

def display_results(img_rgb, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        probabilities = result['probabilities']
        color = tuple(c / 255 for c in EMOTION_COLORS[emotion])
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        # Display the primary emotion with percentage (no background)
        emotion_text = f"{EMOTIONS[emotion]}: {probabilities[emotion]*100:.1f}%"
        plt.text(x, y-10, emotion_text, color=color, fontsize=12)
        
        # Display all emotion percentages below the face (with background)
        all_emotions_text = ""
        for i, prob in enumerate(probabilities):
            all_emotions_text += f"{EMOTIONS[i]}: {prob*100:.1f}%\n"
        
        # Position the detailed percentages below the face
        plt.text(x, y+h+10, all_emotions_text.strip(), fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_results(img_rgb, results, output_path):
    """Save image with emotion annotations"""
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        probabilities = result['probabilities']
        color = tuple(c / 255 for c in EMOTION_COLORS[emotion])
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        # Display the primary emotion with percentage (no background)
        emotion_text = f"{EMOTIONS[emotion]}: {probabilities[emotion]*100:.1f}%"
        plt.text(x, y-10, emotion_text, color=color, fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_frame(frame, model, face_cascade, device):
    """Process a single video frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    results = []
    
    for (x, y, w, h) in faces:
        face_tensor = preprocess_face(frame_rgb[y:y+h, x:x+w])
        emotion, probs = predict_emotion(model, face_tensor, device)
        results.append({'bbox': (x, y, w, h), 'emotion': emotion, 'probabilities': probs})
        
        # Draw rectangle and emotion on the frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add the percentage to the emotion text
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame, results

def apply_previous_results(frame, previous_results):
    """Apply face detection results from a previous frame to the current frame"""
    if not previous_results:
        return frame
    
    for result in previous_results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        probs = result['probabilities']
        color = EMOTION_COLORS[emotion]
        
        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        emotion_text = f"{EMOTIONS[emotion]}: {probs[emotion]*100:.1f}%"
        cv2.putText(frame, emotion_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

def process_video(video_path, model, device, output_path=None, sample_rate=15, display=True):
    """Process video file for emotion detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling - process every Nth frame to achieve sample_rate
    frame_step = max(1, int(fps / sample_rate))
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS")
    print(f"Processing every {frame_step} frames to achieve {sample_rate} samples per second")
    print(f"All frames will be included in the output to maintain original duration")
    
    # Setup face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processed_count = 0
    previous_results = []  # Store the most recent detection results
    
    # Create a tqdm progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Decide whether to process this frame for emotion detection
            if frame_count % frame_step == 0:
                # Process the frame for emotion detection
                processed_frame, results = process_frame(frame.copy(), model, face_cascade, device)
                processed_count += 1
                previous_results = results  # Store results for unprocessed frames
                
                # Add number of detected faces to progress bar description
                pbar.set_postfix({"Detected faces": len(results)})
                
                # Display the processed frame if requested
                if display:
                    cv2.imshow('Emotion Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                
                # Write processed frame to output video
                if out:
                    out.write(processed_frame)
            else:
                # For unprocessed frames, apply the previous results to maintain consistent visualization
                annotated_frame = apply_previous_results(frame.copy(), previous_results)
                
                # Display the frame with previous detection results
                if display:
                    cv2.imshow('Emotion Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                
                # Write the frame with previous detection results
                if out:
                    out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)  # Update progress bar
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    
    finally:
        pbar.close()  # Close the progress bar
        print(f"\nProcessed {processed_count} frames out of {frame_count} total frames")
        print(f"All {frame_count} frames were included in the output video")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

def main():
    model_path = 'models/fer2013_resnet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    # Load the model
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Ask for input type (image or video)
    input_type = input("Do you want to process an image or a video? (image/video): ").strip().lower()
    
    if input_type == "image":
        # Original image processing logic
        image_path = input("Enter the path to your image file: ")
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return
        
        img_rgb, results = detect_and_predict_emotions(image_path, model, DEVICE)
        print(f"Found {len(results)} faces in the image.")
        for i, result in enumerate(results):
            print(f"Face {i+1}: {EMOTIONS[result['emotion']]}")
        
        # Ask if user wants to save the result
        save_option = input("Do you want to save the result? (y/n): ").strip().lower()
        if save_option == 'y':
            output_path = input("Enter the output path (or press Enter for default): ").strip()
            if not output_path:
                output_path = 'output_' + os.path.basename(image_path)
            save_results(img_rgb, results, output_path)
            print(f"Result saved to {output_path}")
        
        display_results(img_rgb, results)
        
    elif input_type == "video":
        # Video processing logic
        video_path = input("Enter the path to your video file: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            return
        
        # Ask for sample rate
        try:
            sample_rate = int(input("Enter sample rate (frames per second, default 15): ") or 15)
        except ValueError:
            sample_rate = 15
            print("Invalid input, using default sample rate of 15 fps")
        
        # Ask if user wants to save the processed video
        save_option = input("Do you want to save the processed video? (y/n): ").strip().lower()
        output_path = None
        if save_option == 'y':
            output_path = input("Enter the output path (or press Enter for default): ").strip()
            if not output_path:
                output_path = 'output_' + os.path.basename(video_path)
        
        # Ask if user wants to display frames while processing
        display_option = input("Display video while processing? (y/n): ").strip().lower()
        display = display_option == 'y'
        
        # Process the video
        process_video(video_path, model, DEVICE, output_path, sample_rate, display)
        print("Video processing complete!")
        
    else:
        print("Invalid input type. Please choose 'image' or 'video'.")

if __name__ == "__main__":
    main()
