import cv2
import numpy as np
import onnxruntime as ort
import os

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

def load_face_model(face_model_path=None):
    """Load ONNX face detection model"""
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
        return None
    
    print(f"Using ONNX face detection model: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    
    return ort_session
