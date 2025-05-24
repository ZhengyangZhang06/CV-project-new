import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels for FER2013
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad", 
    5: "Surprise",
    6: "Neutral"
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

# Create ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def preprocess_image(image_path):
    """
    Preprocess an image for emotion recognition
    """
    # Open and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_emotion(model, image_tensor, device):
    """
    Predict the emotion of a face image
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        # Get probabilities
        probs = F.softmax(outputs, dim=1)
        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item(), probs.cpu().numpy()[0]

def display_results(image_path, emotion, probabilities):
    """
    Display the image and prediction results
    """
    # Load and convert image for display
    image = Image.open(image_path)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Display image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title(f"Predicted Emotion: {EMOTIONS[emotion]}")
    ax1.axis('off')
    
    # Display emotion probabilities as bar chart
    ax2 = fig.add_subplot(1, 2, 2)
    emotions = list(EMOTIONS.values())
    ax2.bar(emotions, probabilities)
    ax2.set_title("Emotion Probabilities")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    # Check if model file exists
    model_path = 'models/fer2013_resnet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please make sure you have trained the model first.")
        return
    
    # Get image path from user
    image_path = input("Enter the path to your image file: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Load the model
    print("Loading model...")
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Preprocess the image
    print("Processing image...")
    image_tensor = preprocess_image(image_path)
    
    # Predict emotion
    print("Predicting emotion...")
    emotion, probabilities = predict_emotion(model, image_tensor, DEVICE)
    
    # Print result
    print(f"Predicted emotion: {EMOTIONS[emotion]}")
    
    # Display results
    display_results(image_path, emotion, probabilities)

if __name__ == "__main__":
    main()