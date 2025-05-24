import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Constants for emotion recognition
IMG_SIZE = 48
EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
#EMOTIONS = {0: "生气", 1: "厌恶", 2: "害怕", 3: "快乐", 4: "悲伤", 5: "惊讶", 6: "中立"}
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

def put_chinese_text(img, text, position, font_size=30, color=(255, 255, 255)):
    """
    在OpenCV图像上显示中文文本的简单方法
    Args:
        img: OpenCV图像 (BGR格式)
        text: 要显示的中文文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (B, G, R)
    """
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)    # 使用Windows系统中文字体，优先选择粗体字体
    try:
        # 常规宋体
        font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", font_size)
    except:
          try:
            # 微软雅黑 ，几乎所有Windows系统都有
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
          except:
            # 如果还是找不到，使用英文显示
            font = ImageFont.load_default()
            # 如果是中文文本但找不到中文字体，改为英文
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                # 包含中文字符，但没有中文字体，使用英文映射
                english_emotions = {
                    "生气": "Angry", "厌恶": "Disgust", "害怕": "Fear", 
                    "快乐": "Happy", "悲伤": "Sad", "惊讶": "Surprise", "中立": "Neutral"
                }
                for chinese, english in english_emotions.items():
                    text = text.replace(chinese, english)
      # 绘制文本 (PIL使用RGB格式，所以要转换颜色)
    rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
    
    # 为了让文字更粗，我们可以绘制多次，稍微偏移位置
    # 这样可以创建粗体效果
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # 跳过中心位置，最后绘制
            draw.text((position[0] + dx, position[1] + dy), text, font=font, fill=rgb_color)
    
    # 最后在中心位置绘制，确保文字清晰
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
