# 🤖 ONNX人脸检测模型使用指南

本文档详细说明了项目中使用的ONNX人脸检测模型的输入输出格式、使用方法和最佳实践。

## 📋 目录
- [模型概述](#模型概述)
- [输入格式详解](#输入格式详解)
- [输出格式详解](#输出格式详解)
- [预处理步骤](#预处理步骤)
- [后处理步骤](#后处理步骤)
- [代码示例](#代码示例)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

## 🎯 模型概述

项目支持多个ONNX人脸检测模型，基于Ultra-Light-Fast-Generic-Face-Detector架构：

| 模型名称 | 输入尺寸 | 文件大小 | 适用场景 |
|---------|---------|---------|---------|
| `version-RFB-320.onnx` | 320×240 | ~1.2MB | 平衡速度与精度 |
| `version-RFB-640.onnx` | 640×480 | ~1.2MB | 高精度检测 |
| `version-slim-320.onnx` | 320×240 | ~0.3MB | 轻量级快速检测 |
| `version-RFB-320_simplified.onnx` | 320×240 | ~1.2MB | 优化版本 |

## 🔍 输入格式详解

### 基本信息
```
输入张量名称: "input"
数据类型: float32
张量形状: [Batch, Channels, Height, Width] (NCHW格式)
```

### 具体规格
```python
# RFB-320 和 Slim-320 模型
input_shape = [1, 3, 240, 320]  # [N, C, H, W]

# RFB-640 模型  
input_shape = [1, 3, 480, 640]  # [N, C, H, W]
```

### 维度说明
- **Batch (N)**: 批次大小，通常为 1
- **Channels (C)**: 颜色通道数，RGB图像为 3
- **Height (H)**: 图像高度
  - 320系列模型: 240像素
  - 640系列模型: 480像素  
- **Width (W)**: 图像宽度
  - 320系列模型: 320像素
  - 640系列模型: 640像素

### 数值范围
- **原始像素值**: [0, 255] (uint8)
- **归一化后**: [-1, 1] (float32)

## 🎯 输出格式详解

模型产生两个输出张量：

### 1. 置信度分数 (scores)
```python
output_name: "scores"
shape: [1, 4420, 2]
dtype: float32
```

**维度解释:**
- `1`: 批次大小
- `4420`: 候选检测框的数量
- `2`: 两个类别的置信度分数
  - 索引 0: 背景置信度
  - 索引 1: 人脸置信度

**数值含义:**
```python
# 对于每个检测框 i
background_confidence = scores[0, i, 0]  # 背景置信度
face_confidence = scores[0, i, 1]        # 人脸置信度

# 通常使用 softmax 概率，和为 1
total_prob = background_confidence + face_confidence ≈ 1.0
```

### 2. 边界框坐标 (boxes)
```python
output_name: "boxes"  
shape: [1, 4420, 4]
dtype: float32
```

**维度解释:**
- `1`: 批次大小
- `4420`: 与置信度对应的检测框数量
- `4`: 边界框坐标 `[x1, y1, x2, y2]`

**坐标格式:**
```python
# 归一化坐标 (0-1 范围)
x1_norm, y1_norm, x2_norm, y2_norm = boxes[0, i, :]

# 转换为实际像素坐标
x1 = x1_norm * image_width
y1 = y1_norm * image_height  
x2 = x2_norm * image_width
y2 = y2_norm * image_height

# 计算边界框宽高
width = x2 - x1
height = y2 - y1
```

## 🔧 预处理步骤

### 1. 图像读取和转换
```python
import cv2
import numpy as np

# 读取图像 (BGR格式)
image_bgr = cv2.imread('image.jpg')

# 转换为RGB格式
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

### 2. 尺寸调整
```python
# 根据模型调整图像尺寸
if model_type == "320":
    target_width, target_height = 320, 240
elif model_type == "640":
    target_width, target_height = 640, 480

# 调整图像尺寸
resized_image = cv2.resize(image_rgb, (target_width, target_height))
```

### 3. 归一化处理
```python
# 像素值归一化到 [-1, 1] 范围
image_mean = np.array([127, 127, 127])  # RGB均值
normalized_image = (resized_image - image_mean) / 128.0

# 转换数据类型
normalized_image = normalized_image.astype(np.float32)
```

### 4. 张量重排
```python
# 从 HWC 转换为 CHW 格式
chw_image = np.transpose(normalized_image, [2, 0, 1])

# 添加批次维度: CHW -> NCHW
input_tensor = np.expand_dims(chw_image, axis=0)

# 最终输入形状: [1, 3, H, W]
print(f"输入张量形状: {input_tensor.shape}")
```

## ⚙️ 后处理步骤

### 1. ONNX推理
```python
import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name

# 运行推理
outputs = session.run(None, {input_name: input_tensor})
confidences, boxes = outputs[0], outputs[1]
```

### 2. 置信度过滤
```python
def filter_faces(confidences, boxes, threshold=0.7):
    """根据置信度过滤人脸检测结果"""
    # 提取人脸置信度 (索引1)
    face_scores = confidences[0, :, 1]
    
    # 应用阈值过滤
    valid_indices = face_scores > threshold
    
    # 筛选有效的框和分数
    filtered_boxes = boxes[0, valid_indices, :]
    filtered_scores = face_scores[valid_indices]
    
    return filtered_boxes, filtered_scores
```

### 3. 坐标转换
```python
def convert_coordinates(boxes, original_width, original_height):
    """将归一化坐标转换为实际像素坐标"""
    # 复制数组避免修改原始数据
    pixel_boxes = boxes.copy()
    
    # 转换 x 坐标
    pixel_boxes[:, [0, 2]] *= original_width   # x1, x2
    
    # 转换 y 坐标  
    pixel_boxes[:, [1, 3]] *= original_height  # y1, y2
    
    # 确保坐标在图像范围内
    pixel_boxes[:, [0, 2]] = np.clip(pixel_boxes[:, [0, 2]], 0, original_width)
    pixel_boxes[:, [1, 3]] = np.clip(pixel_boxes[:, [1, 3]], 0, original_height)
    
    return pixel_boxes.astype(np.int32)
```

### 4. 非极大值抑制 (NMS)
```python
def apply_nms(boxes, scores, iou_threshold=0.3):
    """应用非极大值抑制去除重叠检测框"""
    # 计算面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按置信度排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # 保留IoU小于阈值的框
        order = order[1:][iou <= iou_threshold]
    
    return keep
```

## 💻 完整代码示例

```python
import cv2
import numpy as np
import onnxruntime as ort

def detect_faces(image_path, model_path, confidence_threshold=0.7):
    """完整的人脸检测流程"""
    
    # 1. 加载模型
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # 获取模型输入尺寸
    model_height, model_width = input_shape[2], input_shape[3]
    
    # 2. 读取和预处理图像
    image_bgr = cv2.imread(image_path)
    original_height, original_width = image_bgr.shape[:2]
    
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸
    resized_image = cv2.resize(image_rgb, (model_width, model_height))
    
    # 归一化
    image_mean = np.array([127, 127, 127])
    normalized_image = (resized_image - image_mean) / 128.0
    
    # 重排维度和添加批次
    input_tensor = np.transpose(normalized_image, [2, 0, 1])
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    
    # 3. 推理
    outputs = session.run(None, {input_name: input_tensor})
    confidences, boxes = outputs[0], outputs[1]
    
    # 4. 后处理
    # 过滤低置信度检测
    face_scores = confidences[0, :, 1]
    valid_indices = face_scores > confidence_threshold
    
    filtered_boxes = boxes[0, valid_indices, :]
    filtered_scores = face_scores[valid_indices]
    
    if len(filtered_boxes) == 0:
        return []
    
    # 转换坐标
    pixel_boxes = filtered_boxes.copy()
    pixel_boxes[:, [0, 2]] *= original_width
    pixel_boxes[:, [1, 3]] *= original_height
    pixel_boxes = pixel_boxes.astype(np.int32)
    
    # 应用NMS
    keep_indices = apply_nms(pixel_boxes, filtered_scores)
    
    # 组织结果
    results = []
    for i in keep_indices:
        x1, y1, x2, y2 = pixel_boxes[i]
        confidence = filtered_scores[i]
        results.append({
            'bbox': (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h)
            'confidence': float(confidence),
            'coordinates': (x1, y1, x2, y2)       # (x1, y1, x2, y2)
        })
    
    return results

# 使用示例
if __name__ == "__main__":
    results = detect_faces(
        image_path="test_image.jpg",
        model_path="models/version-RFB-320.onnx",
        confidence_threshold=0.7
    )
    
    print(f"检测到 {len(results)} 个人脸:")
    for i, result in enumerate(results):
        print(f"人脸 {i+1}: 边界框={result['bbox']}, 置信度={result['confidence']:.3f}")
```

## 📊 性能对比

| 模型 | 输入尺寸 | 推理时间* | 内存占用 | 检测精度 | 推荐用途 |
|------|---------|-----------|----------|----------|----------|
| RFB-320 | 320×240 | ~15ms | ~50MB | ⭐⭐⭐⭐ | 通用场景 |
| RFB-640 | 640×480 | ~45ms | ~120MB | ⭐⭐⭐⭐⭐ | 高精度要求 |
| Slim-320 | 320×240 | ~8ms | ~30MB | ⭐⭐⭐ | 实时应用 |
| RFB-320_simplified | 320×240 | ~12ms | ~45MB | ⭐⭐⭐⭐ | 优化场景 |

*推理时间基于CPU测试，实际性能因硬件而异

## ❓ 常见问题

### Q1: 为什么检测不到人脸？
**可能原因:**
- 置信度阈值过高，尝试降低到 0.5-0.6
- 输入图像尺寸过小，人脸像素太少
- 图像质量差，模糊或光照不佳

**解决方案:**
```python
# 降低置信度阈值
results = detect_faces(image_path, model_path, confidence_threshold=0.5)

# 检查输入图像尺寸
print(f"图像尺寸: {image.shape}")
```

### Q2: 检测结果有重复框？
**原因:** NMS阈值过高或未应用NMS

**解决方案:**
```python
# 降低NMS IoU阈值
keep_indices = apply_nms(boxes, scores, iou_threshold=0.3)
```

### Q3: 坐标转换错误？
**原因:** 归一化坐标未正确转换

**检查代码:**
```python
# 确保使用原始图像尺寸，不是模型输入尺寸
pixel_boxes[:, [0, 2]] *= original_width   # 不是 model_width
pixel_boxes[:, [1, 3]] *= original_height  # 不是 model_height
```

### Q4: 如何选择合适的模型？
**选择指南:**
- **实时应用**: Slim-320 (速度优先)
- **一般用途**: RFB-320 (平衡性能)
- **高精度**: RFB-640 (精度优先)
- **优化部署**: RFB-320_simplified (优化版本)

### Q5: 批处理如何实现？
**批处理示例:**
```python
# 修改输入维度支持批处理
batch_size = 4
input_tensor = np.zeros((batch_size, 3, 240, 320), dtype=np.float32)

for i, image in enumerate(batch_images):
    # 预处理每张图像
    processed = preprocess_image(image)
    input_tensor[i] = processed

# 批量推理
outputs = session.run(None, {input_name: input_tensor})
```

## 📚 参考资料

- [Ultra-Light-Fast-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- [ONNX Runtime 官方文档](https://onnxruntime.ai/)
- [OpenCV 图像处理文档](https://docs.opencv.org/)

---

**最后更新:** 2025年5月24日  
**版本:** 1.0.0  
**维护者:** CV-project-new 团队
