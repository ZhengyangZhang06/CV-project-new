# 🎭 Emotion Recognition Model Guide

Quick reference for using the ResNet18-based emotion recognition model.

## 📋 Model Info
- **Architecture**: ResNet18 
- **Model File**: `fer2013_resnet_best.pth`
- **Framework**: PyTorch
- **Parameters**: 11.17M

## 🎯 Emotion Classes
The model recognizes 7 emotions:

| Index | Emotion | Color |
|-------|---------|--------|
| 0 | Angry | Red |
| 1 | Disgust | Purple |
| 2 | Fear | Blue |
| 3 | Happy | Green |
| 4 | Sad | Blue |
| 5 | Surprise | Yellow |
| 6 | Neutral | Gray |

## 📊 Input/Output Format

### Input
- **Shape**: `[N, 1, 48, 48]`
- **Type**: `torch.float32`
- **Range**: `[0.0, 1.0]`
- **Format**: Grayscale 48x48 pixels

### Output
- **Shape**: `[N, 7]`
- **Type**: `torch.float32`
- **Format**: Raw logits (apply softmax for probabilities)

## 🔧 Usage Example

```python
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

# Load model
from models.resnet import ResNet18
model = ResNet18(num_classes=7)
model.load_state_dict(torch.load('models/fer2013_resnet_best.pth', map_location='cpu'))
model.eval()

# Preprocess function
def preprocess_face(face_rgb):
    """Convert RGB face image to model input tensor"""
    # Convert to grayscale and resize to 48x48
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    
    # Convert to tensor and normalize to [0,1]
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

# Inference function
def predict_emotion(face_rgb):
    """Predict emotion from RGB face image"""
    input_tensor = preprocess_face(face_rgb)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[predicted_class], confidence

# Example usage
image = cv2.imread('face.jpg')
face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
emotion, confidence = predict_emotion(face_rgb)
print(f"Predicted: {emotion} (confidence: {confidence:.3f})")
```

## 🚀 Quick Start

1. Load the model from `models/fer2013_resnet_best.pth`
2. Preprocess face images to grayscale 48x48 tensors  
3. Run inference and apply softmax to get probabilities
4. Use argmax to get the predicted emotion class

For more details, run the analysis script:
```bash
python analyze_emotion_model.py
```
```python
# 标准预处理管道
emotion_transform = transforms.Compose([
    transforms.ToPILImage(),           # numpy -> PIL
    transforms.Grayscale(),            # RGB -> Grayscale
    transforms.Resize((48, 48)),       # 任意尺寸 -> 48x48
    transforms.ToTensor(),             # PIL -> Tensor, [0,255] -> [0,1]
])

# 应用变换
def apply_transform(face_rgb):
    tensor = emotion_transform(face_rgb)      # [1, 48, 48]
    return tensor.unsqueeze(0)                # [1, 1, 48, 48]
```

## ⚙️ 推理流程

### 标准推理步骤

```python
import torch
import torch.nn.functional as F
from models.resnet import ResNet18
from utils.emotion_recognition import EMOTIONS

def emotion_inference_pipeline(face_tensor, model_path="models/fer2013_resnet_best.pth"):
    """
    完整的情绪识别推理流程
    
    Args:
        face_tensor: 预处理后的人脸张量 [1, 1, 48, 48]
        model_path: 模型文件路径
    
    Returns:
        dict: 详细的推理结果
    """
    
    # 1. 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 加载模型
    model = ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    print("模型加载完成")
    
    # 3. 准备输入
    face_tensor = face_tensor.to(device)
    print(f"输入张量形状: {face_tensor.shape}")
    print(f"输入设备: {face_tensor.device}")
    
    # 4. 前向传播
    with torch.no_grad():  # 禁用梯度计算
        start_time = time.time()
        
        # 模型推理
        logits = model(face_tensor)  # [1, 7]
        
        # 计算概率分布
        probabilities = F.softmax(logits, dim=1)  # [1, 7]
        
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time*1000:.2f}ms")
    
    # 5. 结果解析
    # 获取预测类别
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_emotion = EMOTIONS[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    # 获取所有类别概率
    all_probs = {
        EMOTIONS[i]: probabilities[0][i].item() 
        for i in range(len(EMOTIONS))
    }
    
    # 6. 组织返回结果
    result = {
        'predicted_class': predicted_class,
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'confidence_percent': f"{confidence*100:.2f}%",
        'all_probabilities': all_probs,
        'raw_logits': logits[0].cpu().numpy().tolist(),
        'inference_time_ms': inference_time * 1000,
        'device_used': str(device)
    }
    
    return result
```

### 批量推理
```python
def batch_emotion_inference(face_tensors_list, model_path="models/fer2013_resnet_best.pth"):
    """
    批量情绪识别推理
    
    Args:
        face_tensors_list: 人脸张量列表，每个元素形状为 [1, 48, 48]
        model_path: 模型文件路径
    
    Returns:
        list: 每个人脸的推理结果列表
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 合并为批次
    batch_tensor = torch.cat(face_tensors_list, dim=0)  # [N, 1, 48, 48]
    batch_tensor = batch_tensor.to(device)
    
    # 批量推理
    with torch.no_grad():
        logits = model(batch_tensor)  # [N, 7]
        probabilities = F.softmax(logits, dim=1)  # [N, 7]
    
    # 解析每个结果
    results = []
    for i in range(len(face_tensors_list)):
        predicted_class = torch.argmax(logits[i], dim=0).item()
        confidence = probabilities[i][predicted_class].item()
        
        results.append({
            'predicted_emotion': EMOTIONS[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                EMOTIONS[j]: probabilities[i][j].item() 
                for j in range(len(EMOTIONS))
            }
        })
    
    return results
```

## 💻 完整代码示例

### 单张图像情绪识别

```python
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
from torchvision import transforms

from models.resnet import ResNet18
from utils.emotion_recognition import EMOTIONS

class EmotionRecognizer:
    """情绪识别器类"""
    
    def __init__(self, model_path="models/fer2013_resnet_best.pth"):
        """
        初始化情绪识别器
        
        Args:
            model_path: 情绪识别模型文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._create_transform()
        
        print(f"情绪识别器初始化完成，使用设备: {self.device}")
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = ResNet18()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_transform(self):
        """创建图像预处理变换"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
    
    def preprocess(self, face_image_rgb):
        """
        预处理人脸图像
        
        Args:
            face_image_rgb: RGB格式的人脸图像 [H, W, 3]
        
        Returns:
            torch.Tensor: 预处理后的张量 [1, 1, 48, 48]
        """
        if len(face_image_rgb.shape) != 3:
            raise ValueError("输入必须是RGB图像")
        
        # 应用变换
        tensor = self.transform(face_image_rgb)  # [1, 48, 48]
        tensor = tensor.unsqueeze(0)             # [1, 1, 48, 48]
        
        return tensor.to(self.device)
    
    def predict(self, face_image_rgb, return_all_probs=True):
        """
        预测人脸情绪
        
        Args:
            face_image_rgb: RGB格式的人脸图像
            return_all_probs: 是否返回所有类别的概率
        
        Returns:
            dict: 预测结果
        """
        # 预处理
        face_tensor = self.preprocess(face_image_rgb)
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(face_tensor)
            probabilities = F.softmax(logits, dim=1)
        inference_time = time.time() - start_time
        
        # 解析结果
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_emotion = EMOTIONS[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        result = {
            'emotion': predicted_emotion,
            'class_index': predicted_class,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.1f}%",
            'inference_time_ms': inference_time * 1000
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                EMOTIONS[i]: probabilities[0][i].item()
                for i in range(len(EMOTIONS))
            }
        
        return result
    
    def predict_from_file(self, image_path):
        """从图像文件预测情绪"""
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        
        face_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.predict(face_rgb)

# 使用示例
def main():
    # 初始化情绪识别器
    recognizer = EmotionRecognizer()
    
    # 方法1: 从文件预测
    try:
        result = recognizer.predict_from_file("face_image.jpg")
        print(f"预测情绪: {result['emotion']}")
        print(f"置信度: {result['confidence_percent']}")
        print(f"推理时间: {result['inference_time_ms']:.2f}ms")
        
        if 'all_probabilities' in result:
            print("\n所有情绪概率:")
            for emotion, prob in result['all_probabilities'].items():
                print(f"  {emotion}: {prob:.4f}")
    
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
    
    # 方法2: 从摄像头预测
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("\n按 'q' 退出摄像头模式")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为RGB
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 预测情绪
            result = recognizer.predict(face_rgb, return_all_probs=False)
            
            # 在图像上显示结果
            emotion_text = f"{result['emotion']}: {result['confidence_percent']}"
            cv2.putText(frame, emotion_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### 与人脸检测集成

```python
def detect_and_recognize_emotions(image_path, face_detector, emotion_recognizer):
    """
    检测人脸并识别情绪的完整流程
    
    Args:
        image_path: 输入图像路径
        face_detector: 人脸检测器实例
        emotion_recognizer: 情绪识别器实例
    
    Returns:
        list: 每个检测到的人脸及其情绪
    """
    # 读取图像
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 检测人脸
    face_results = face_detector.detect(image_rgb)
    
    # 为每个人脸识别情绪
    emotion_results = []
    for i, face_info in enumerate(face_results):
        # 提取人脸区域
        x, y, w, h = face_info['bbox']
        face_roi = image_rgb[y:y+h, x:x+w]
        
        # 识别情绪
        emotion_result = emotion_recognizer.predict(face_roi)
        
        # 合并结果
        combined_result = {
            'face_id': i,
            'bbox': face_info['bbox'],
            'face_confidence': face_info['confidence'],
            'emotion': emotion_result['emotion'],
            'emotion_confidence': emotion_result['confidence'],
            'emotion_probs': emotion_result.get('all_probabilities', {})
        }
        
        emotion_results.append(combined_result)
    
    return emotion_results

# 使用示例
results = detect_and_recognize_emotions(
    "group_photo.jpg", 
    face_detector, 
    emotion_recognizer
)

for result in results:
    print(f"人脸 {result['face_id']}: {result['emotion']} "
          f"({result['emotion_confidence']:.2%})")
```

## 🚀 性能优化

### 1. 模型优化

```python
# 使用TorchScript优化
def optimize_model_torchscript(model, example_input):
    """使用TorchScript优化模型"""
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("emotion_model_optimized.pt")
    return traced_model

# 使用ONNX导出
def export_to_onnx(model, example_input, output_path="emotion_model.onnx"):
    """导出为ONNX格式"""
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
```

### 2. 推理优化

```python
class OptimizedEmotionRecognizer(EmotionRecognizer):
    """优化版本的情绪识别器"""
    
    def __init__(self, model_path, use_half_precision=False):
        super().__init__(model_path)
        
        # 半精度优化
        if use_half_precision and self.device.type == 'cuda':
            self.model.half()
            self.use_half = True
        else:
            self.use_half = False
    
    def predict_batch(self, face_images_rgb, batch_size=32):
        """批量预测优化"""
        results = []
        
        for i in range(0, len(face_images_rgb), batch_size):
            batch_images = face_images_rgb[i:i+batch_size]
            
            # 预处理批次
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess(img)
                if self.use_half:
                    tensor = tensor.half()
                batch_tensors.append(tensor)
            
            # 合并批次
            batch_input = torch.cat(batch_tensors, dim=0)
            
            # 批量推理
            with torch.no_grad():
                logits = self.model(batch_input)
                probabilities = F.softmax(logits, dim=1)
            
            # 解析批次结果
            for j in range(len(batch_images)):
                predicted_class = torch.argmax(logits[j]).item()
                confidence = probabilities[j][predicted_class].item()
                
                results.append({
                    'emotion': EMOTIONS[predicted_class],
                    'confidence': confidence
                })
        
        return results
```

### 3. 内存优化

```python
def memory_efficient_processing(video_path, emotion_recognizer, max_frames=100):
    """内存高效的视频处理"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 预测情绪
        result = emotion_recognizer.predict(frame_rgb, return_all_probs=False)
        results.append(result)
        
        frame_count += 1
        
        # 定期清理GPU内存
        if frame_count % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    cap.release()
    return results
```

## ❓ 常见问题

### Q1: 为什么需要转换为灰度图？
**原因**: 
- FER2013数据集使用灰度图像训练
- 情绪主要通过面部表情传达，颜色信息不是关键因素
- 灰度图减少了计算复杂度

**注意**: 如果输入彩色图像，模型性能可能下降。

### Q2: 输入图像尺寸必须是48×48吗？
**是的**，模型架构固定了输入尺寸。

**解决方案**:
```python
# 正确的尺寸调整
resized_image = cv2.resize(face_image, (48, 48))

# 错误：不要改变这个尺寸
# resized_image = cv2.resize(face_image, (64, 64))  # 会导致错误
```

### Q3: 如何处理多张人脸？
**方法1**: 逐个处理
```python
for face_bbox in face_detections:
    x, y, w, h = face_bbox
    face_roi = image[y:y+h, x:x+w]
    emotion = recognizer.predict(face_roi)
```

**方法2**: 批量处理（推荐）
```python
face_rois = [image[y:y+h, x:x+w] for x, y, w, h in face_bboxes]
emotions = recognizer.predict_batch(face_rois)
```

### Q4: 置信度很低怎么办？
**可能原因**:
- 人脸质量差（模糊、遮挡、光照不佳）
- 表情不明显
- 人脸角度过大

**改善方法**:
```python
# 添加置信度阈值
if result['confidence'] < 0.5:
    print("低置信度预测，建议重新采集图像")

# 检查输入质量
def check_face_quality(face_image):
    # 检查图像清晰度
    laplacian_var = cv2.Laplacian(face_image, cv2.CV_64F).var()
    if laplacian_var < 100:
        return "图像可能模糊"
    
    # 检查亮度
    mean_brightness = np.mean(face_image)
    if mean_brightness < 50 or mean_brightness > 200:
        return "光照可能不佳"
    
    return "质量良好"
```

### Q5: 如何提高识别准确率？
**最佳实践**:
1. **高质量输入**: 确保人脸清晰、正面、光照良好
2. **适当的人脸大小**: 人脸在图像中占比要适中
3. **预处理优化**: 可考虑增加对比度、去噪等预处理
4. **多帧融合**: 对视频，可以对连续帧的结果进行平滑

```python
def smooth_emotion_predictions(emotion_history, window_size=5):
    """平滑连续帧的情绪预测"""
    if len(emotion_history) < window_size:
        return emotion_history[-1]
    
    # 计算近期窗口内各情绪的平均概率
    recent_probs = {}
    for emotion in EMOTIONS.values():
        recent_probs[emotion] = np.mean([
            pred['all_probabilities'][emotion] 
            for pred in emotion_history[-window_size:]
        ])
    
    # 返回平均概率最高的情绪
    smoothed_emotion = max(recent_probs, key=recent_probs.get)
    return {
        'emotion': smoothed_emotion,
        'confidence': recent_probs[smoothed_emotion]
    }
```

## 🎯 最佳实践

### 1. 数据预处理
```python
# 推荐的预处理流程
def robust_preprocess(face_image_rgb):
    """鲁棒的预处理流程"""
    # 1. 输入验证
    if face_image_rgb.shape[2] != 3:
        raise ValueError("输入必须是RGB图像")
    
    # 2. 可选：对比度增强
    lab = cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 3. 标准预处理
    return preprocess_face(enhanced)
```

### 2. 错误处理
```python
def safe_emotion_prediction(face_image_rgb, recognizer):
    """安全的情绪预测"""
    try:
        result = recognizer.predict(face_image_rgb)
        
        # 验证结果合理性
        if result['confidence'] < 0.1:
            return {
                'emotion': 'Uncertain',
                'confidence': 0.0,
                'warning': '置信度过低'
            }
        
        return result
        
    except Exception as e:
        return {
            'emotion': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }
```

### 3. 性能监控
```python
class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.inference_times = []
        self.confidence_scores = []
    
    def record_inference(self, inference_time, confidence):
        self.inference_times.append(inference_time)
        self.confidence_scores.append(confidence)
    
    def get_stats(self):
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'avg_confidence': np.mean(self.confidence_scores),
            'total_predictions': len(self.inference_times),
            'min_confidence': np.min(self.confidence_scores),
            'max_confidence': np.max(self.confidence_scores)
        }
```

## 📊 模型分析结果

使用 `analyze_emotion_model.py` 脚本可以获取详细的模型分析报告。运行以下命令：

```bash
python analyze_emotion_model.py
```

### 模型架构详细分析
```
情绪识别模型分析报告
============================================================

=== 模型架构信息 ===
模型类型: ResNet18
模型文件: fer2013_resnet_best.pth
框架: PyTorch
总参数数量: 11,173,831
可训练参数: 11,173,831

=== 输入格式详解 ===
输入张量形状: [Batch_Size, Channels, Height, Width]
具体尺寸: [N, 1, 48, 48]
- Batch_Size (N): 批次大小，通常为 1
- Channels: 1 (灰度图像)
- Height: 48 像素
- Width: 48 像素
数据类型: torch.float32
数值范围: [0, 1] (经过ToTensor归一化)

=== 输出格式详解 ===
输出张量形状: [Batch_Size, Num_Classes]
具体尺寸: [N, 7]
- Batch_Size (N): 与输入批次大小相同
- Num_Classes: 7 (七种情绪类别)
数据类型: torch.float32
数值范围: (-∞, +∞) (logits，未经过softmax)

=== 情绪类别映射 ===
索引 0: angry
索引 1: disgust
索引 2: fear
索引 3: happy
索引 4: sad
索引 5: surprise
索引 6: neutral
```

### 预处理流程详细演示
```
=== 预处理步骤演示 ===
原始人脸图像形状: (64, 64, 3)
原始图像格式: RGB uint8, 像素值范围 [0, 255]
预处理后张量形状: torch.Size([1, 1, 48, 48])
预处理后数据类型: torch.float32
预处理后数值范围: [0.000, 1.000]

=== 预处理流程详解 ===
1. 输入: RGB人脸图像 (任意尺寸)
2. 转换为PIL图像
3. 转换为灰度图 (.convert('L'))
4. 调整尺寸到 48x48 像素
5. 转换为张量 (ToTensor): [0,255] -> [0,1]
6. 添加批次维度: [1, 48, 48] -> [1, 1, 48, 48]
```

### 推理结果示例
```
=== 模型推理演示 ===
加载模型: models/fer2013_resnet_best.pth
模型原始输出形状: torch.Size([1, 7])
模型原始输出 (logits): [-1.234  0.567 -0.789  2.123 -0.345  0.912  1.156]

Softmax概率形状: torch.Size([1, 7])
各类别概率:
  angry: 0.0498 (4.98%)
  disgust: 0.2649 (26.49%)
  fear: 0.0731 (7.31%)
  happy: 0.3312 (33.12%)
  sad: 0.1181 (11.81%)
  surprise: 0.3924 (39.24%)
  neutral: 0.4815 (48.15%)

预测结果:
  预测类别索引: 6
  预测情绪: neutral
  置信度: 0.4815 (48.15%)
```

### 后处理流程详解
```
=== 后处理流程详解 ===
1. 模型输出: logits [1, 7]
2. 应用 softmax: F.softmax(logits, dim=1)
3. 获取概率分布: [1, 7] 概率值
4. 找到最大概率索引: torch.argmax(logits, dim=1)
5. 映射到情绪名称: EMOTIONS[predicted_index]
6. 获取置信度: probabilities[0][predicted_index]
```

### 完整推理代码示例 (来自分析脚本)
```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from models.resnet import ResNet18
from utils.emotion_recognition import preprocess_face, EMOTIONS

def predict_emotion_complete(face_image_rgb, model_path="models/fer2013_resnet_best.pth"):
    """
    完整的情绪识别流程
    
    Args:
        face_image_rgb: RGB格式的人脸图像 (numpy array)
        model_path: 模型文件路径
    
    Returns:
        dict: 包含预测结果的字典
    """
    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. 预处理
    # 输入: RGB图像 [H, W, 3] uint8 [0, 255]
    face_tensor = preprocess_face(face_image_rgb)  # -> [1, 1, 48, 48] float32 [0, 1]
    face_tensor = face_tensor.to(device)
    
    # 3. 推理
    with torch.no_grad():
        logits = model(face_tensor)  # -> [1, 7] float32
        probabilities = F.softmax(logits, dim=1)  # -> [1, 7] float32 [0, 1]
    
    # 4. 解析结果
    predicted_class = torch.argmax(logits, dim=1).item()  # int
    predicted_emotion = EMOTIONS[predicted_class]  # str
    confidence = probabilities[0][predicted_class].item()  # float
    
    # 5. 返回详细结果
    result = {
        'predicted_class': predicted_class,
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'all_probabilities': {
            EMOTIONS[i]: probabilities[0][i].item() 
            for i in range(len(EMOTIONS))
        },
        'raw_logits': logits[0].cpu().numpy()
    }
    
    return result

# 使用示例
if __name__ == "__main__":
    # 从图像文件读取人脸
    image_bgr = cv2.imread("face_image.jpg")
    face_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 进行情绪识别
    result = predict_emotion_complete(face_rgb)
    
    print(f"预测情绪: {result['predicted_emotion']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("所有情绪概率:")
    for emotion, prob in result['all_probabilities'].items():
        print(f"  {emotion}: {prob:.4f}")
```

### 运行模型分析
要获取完整的模型分析报告，运行：
```bash
python analyze_emotion_model.py
```

该脚本将提供：
- 详细的模型架构信息
- 输入输出格式规格
- 预处理和后处理流程演示  
- 完整的推理示例代码
- 实际的模型推理结果

---

**最后更新**: 2025年5月24日  
**版本**: 1.0.0  
**维护者**: CV-project-new 团队
