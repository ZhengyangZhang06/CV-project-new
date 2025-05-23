import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.face_detection import load_face_model

def draw_grid_info(image, grid_w, grid_h, frame_w, frame_h, model_w, model_h):
    """Draw grid information on the image"""
    # Create a copy of the image
    img_info = image.copy()
    
    # Draw the inner grid lines
    # Vertical line in the middle
    cv2.line(img_info, (grid_w//2, 0), (grid_w//2, grid_h), (255, 255, 255), 2)
    # Horizontal line in the middle
    cv2.line(img_info, (0, grid_h//2), (grid_w, grid_h//2), (255, 255, 255), 2)
    
    # Add frame dimension labels
    for i in range(2):
        for j in range(2):
            x = j * frame_w + frame_w // 2
            y = i * frame_h + frame_h // 2
            cv2.putText(img_info, f"Frame {i*2+j+1}", (x-70, y-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(img_info, f"{frame_w}x{frame_h}", (x-70, y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add grid dimensions in the corner
    cv2.putText(img_info, f"Grid size: {grid_w}x{grid_h}", (10, grid_h-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add model input dimensions
    cv2.putText(img_info, f"Model input: {model_w}x{model_h}", (10, grid_h-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_info

def visualize_grid_details(video_path, face_model_path=None, output_path="output/grid_details.png"):
    """Generate a detailed visualization of the grid transformation process"""
    # Load face model to get dimensions
    result = load_face_model(face_model_path)
    if not result:
        return
    
    # Unpack model dimensions
    _, model_width, model_height = result
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create an empty figure with multiple subplots
    plt.figure(figsize=(18, 10))
    
    # Read up to 4 frames
    frames = []
    for i in range(4):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
        
        # Display frame
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {i+1} ({frame_width}x{frame_height})")
        plt.axis('off')
    
    cap.release()
    
    # Create 2x2 grid with all frames
    grid_h = 2 * frame_height
    grid_w = 2 * frame_width
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill grid with available frames
    for i, frame in enumerate(frames):
        if i >= 4:
            break
        row, col = divmod(i, 2)
        y_offset = row * frame_height
        x_offset = col * frame_width
        grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
    
    # Draw grid information
    grid_with_info = draw_grid_info(grid_image, grid_w, grid_h, 
                                   frame_width, frame_height,
                                   model_width, model_height)
    
    # Display annotated grid
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(grid_with_info, cv2.COLOR_BGR2RGB))
    plt.title("2x2 Grid with Information")
    plt.axis('off')
    
    # Display resized grid for model input
    resized_grid = cv2.resize(grid_image, (model_width, model_height))
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(resized_grid, cv2.COLOR_BGR2RGB))
    plt.title(f"Resized for Model ({model_width}x{model_height})")
    plt.axis('off')
    
    # Add overall title
    plt.suptitle("Batch Processing Visualization", fontsize=16)
    plt.tight_layout()
      # Save or show the figure
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        output_path = "output/grid_details.png"
        os.makedirs("output", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate detailed grid visualization')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--face_model', help='Path to ONNX face detection model')
    parser.add_argument('--output', default='output/grid_details.png', help='Output image path')
    args = parser.parse_args()
    
    visualize_grid_details(args.video, args.face_model, args.output)
