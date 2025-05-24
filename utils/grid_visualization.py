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

def get_optimal_grid_dimensions(batch_size, frame_width=None, frame_height=None):
    """Get optimal grid dimensions for different batch sizes based on frame aspect ratio
    
    Args:
        batch_size: Number of frames to arrange
        frame_width: Width of individual frames (optional)
        frame_height: Height of individual frames (optional)
    
    Returns:
        tuple: (rows, cols) for the grid
    """
    if batch_size == 1:
        return (1, 1)
    elif batch_size == 2:
        # Choose layout based on frame aspect ratio
        if frame_width and frame_height:
            aspect_ratio = frame_width / frame_height
            if aspect_ratio > 1.0:  # Wide frames
                return (1, 2)  # Horizontal layout
            else:  # Tall frames
                return (2, 1)  # Vertical layout
        return (1, 2)  # Default horizontal
    elif batch_size <= 4:
        return (2, 2)
    elif batch_size == 6:
        # Choose layout based on frame aspect ratio
        if frame_width and frame_height:
            aspect_ratio = frame_width / frame_height
            if aspect_ratio > 1.0:  # Wide frames
                return (2, 3)  # 2 rows, 3 cols
            else:  # Tall frames
                return (3, 2)  # 3 rows, 2 cols
        return (2, 3)  # Default
    elif batch_size <= 9:
        return (3, 3)
    elif batch_size <= 16:
        return (4, 4)
    elif batch_size <= 25:
        return (5, 5)
    elif batch_size <= 36:
        return (6, 6)
    else:
        # For larger batches, use square grid
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        return (grid_size, grid_size)

def get_grid_dimensions(batch_size):
    """Legacy function for backward compatibility"""
    return get_optimal_grid_dimensions(batch_size)

def visualize_grid(frames, titles=None, grid_shape=None, figsize=None, suptitle=None):
    """Visualize a grid of frames with titles and dimensions, automatically adapting to batch size
    
    Args:
        frames: List of frames to display
        titles: Optional list of titles for each frame
        grid_shape: Optional tuple (rows, cols) for grid layout
        figsize: Optional tuple (width, height) for figure size
        suptitle: Optional main title for the entire figure
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    num_frames = len(frames)
    
    # Auto-determine grid shape if not provided
    if grid_shape is None:
        frame_width = frames[0].shape[1] if frames else None
        frame_height = frames[0].shape[0] if frames else None
        grid_shape = get_optimal_grid_dimensions(num_frames, frame_width, frame_height)
    
    # Auto-determine figure size if not provided
    if figsize is None:
        base_size = 4
        figsize = (grid_shape[1] * base_size, grid_shape[0] * base_size)
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    
    # Handle single subplot case
    if grid_shape[0] * grid_shape[1] == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Display frames
    for i in range(grid_shape[0] * grid_shape[1]):
        ax = axes[i]
        
        if i < len(frames):
            frame = frames[i]
            # Convert to RGB for matplotlib
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
            else:
                ax.imshow(frame, cmap='gray')
            
            # Add title with dimensions if available
            title = f"Frame {i+1}"
            if titles and i < len(titles):
                title = titles[i]
            h, w = frame.shape[:2]
            title += f" ({w}x{h})"
            ax.set_title(title)
        else:
            # Empty subplot for unused grid positions
            ax.set_visible(False)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    return fig

def create_batch_grid(frames, grid_rows, grid_cols):
    """Create a batch grid image from individual frames
    
    Args:
        frames: List of frames to combine
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
    
    Returns:
        numpy.ndarray: Combined grid image
    """
    if not frames:
        return None
    
    frame_height, frame_width = frames[0].shape[:2]
    grid_h = grid_rows * frame_height
    grid_w = grid_cols * frame_width
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill the grid with frames
    for i, frame in enumerate(frames):
        if i >= grid_rows * grid_cols:
            break
        row, col = divmod(i, grid_cols)
        y_offset = row * frame_height
        x_offset = col * frame_width
        grid_image[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
    
    return grid_image

def add_grid_lines(image, grid_rows, grid_cols, frame_height, frame_width, color=(255, 255, 255), thickness=2, alpha=0.7):
    """Add grid lines to show frame boundaries
    
    Args:
        image: Image to draw on (matplotlib axes or OpenCV image)
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        frame_height: Height of each frame
        frame_width: Width of each frame
        color: Line color
        thickness: Line thickness
        alpha: Line transparency (for matplotlib)
    
    Returns:
        None (modifies image in place)
    """
    if hasattr(image, 'axhline'):  # matplotlib axes
        for i in range(1, grid_rows):
            image.axhline(y=i*frame_height, color=[c/255.0 for c in color], linewidth=thickness, alpha=alpha)
        for i in range(1, grid_cols):
            image.axvline(x=i*frame_width, color=[c/255.0 for c in color], linewidth=thickness, alpha=alpha)
    else:  # OpenCV image
        for i in range(1, grid_rows):
            cv2.line(image, (0, i*frame_height), (grid_cols*frame_width, i*frame_height), color, thickness)
        for i in range(1, grid_cols):
            cv2.line(image, (i*frame_width, 0), (i*frame_width, grid_rows*frame_height), color, thickness)

def map_face_to_frame(face_box, grid_cols, frame_width, frame_height):
    """Map face coordinates from grid back to individual frame
    
    Args:
        face_box: Face bounding box (x1, y1, x2, y2) in grid coordinates
        grid_cols: Number of columns in the grid
        frame_width: Width of each frame
        frame_height: Height of each frame
    
    Returns:
        tuple: (frame_idx, local_bbox) where local_bbox is (x1, y1, x2, y2) in frame coordinates
    """
    x1, y1, x2, y2 = face_box
    
    # Determine which frame this face belongs to
    frame_row = int(y1 // frame_height)
    frame_col = int(x1 // frame_width)
    frame_idx = frame_row * grid_cols + frame_col
    
    # Adjust coordinates to the original frame
    x1_local = x1 - (frame_col * frame_width)
    y1_local = y1 - (frame_row * frame_height)
    x2_local = x2 - (frame_col * frame_width)
    y2_local = y2 - (frame_row * frame_height)
    
    # Make sure coordinates are within the frame bounds
    x1_local = max(0, min(x1_local, frame_width))
    y1_local = max(0, min(y1_local, frame_height))
    x2_local = max(0, min(x2_local, frame_width))
    y2_local = max(0, min(y2_local, frame_height))
    
    return frame_idx, (x1_local, y1_local, x2_local, y2_local)

def save_visualization(fig, output_path, filename, dpi=150):
    """Save visualization figure to file
    
    Args:
        fig: matplotlib figure to save
        output_path: Directory to save to
        filename: Name of the file
        dpi: Resolution for saving
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {full_path}")
    else:
        output_path = "output"
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {full_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate detailed grid visualization')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--face_model', help='Path to ONNX face detection model')
    parser.add_argument('--output', default='output/grid_details.png', help='Output image path')
    args = parser.parse_args()
    
    visualize_grid_details(args.video, args.face_model, args.output)
