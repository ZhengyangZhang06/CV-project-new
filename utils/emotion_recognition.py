import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Constants for emotion recognition
IMG_SIZE = 48
EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
EMOTION_COLORS = {
    0: (255, 0, 0), 1: (128, 0, 128), 2: (128, 0, 255), 3: (0, 255, 0),
    4: (0, 0, 255), 5: (255, 255, 0), 6: (192, 192, 192)
}

def preprocess_face(face_img):
    """Convert an image to the format required for the emotion model"""
    # Convert to grayscale and resize
    face_pil = Image.fromarray(face_img).convert('L')
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    return transform(face_pil).unsqueeze(0)

def predict_emotion(model, face_tensor, device):
    """Predict emotion from a preprocessed face tensor"""
    model.eval()
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probs.cpu().numpy()[0]
