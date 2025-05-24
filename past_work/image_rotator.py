import cv2
import numpy as np
import argparse
from pathlib import Path

def rotate_image(image, angle, center=None, scale=1.0):
    """
    Rotate an image by a specified angle.
    
    Parameters:
    - image: The input image
    - angle: Rotation angle in degrees (positive for counter-clockwise)
    - center: Center of rotation (default is center of the image)
    - scale: Scaling factor (default is 1.0, no scaling)
    
    Returns:
    - The rotated image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # If center is not specified, use the center of the image
    if center is None:
        center = (width // 2, height // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Calculate the new image dimensions to ensure the entire rotated image is visible
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust the rotation matrix to take into account the new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
    
    return rotated_image

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Rotate an image to an arbitrary angle.')
    parser.add_argument('input_image', type=str, help='Path to the input image', default='jg.png')
    parser.add_argument('output_image', type=str, help='Path to save the rotated image', default='jg_r45.png')
    parser.add_argument('angle', type=float, help='Rotation angle in degrees (positive for counter-clockwise)', default=45.0)
    parser.add_argument('--center_x', type=int, default=None, help='X-coordinate of rotation center')
    parser.add_argument('--center_y', type=int, default=None, help='Y-coordinate of rotation center')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling factor')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image '{args.input_image}' does not exist.")
        return
    
    # Load the image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Failed to load image '{args.input_image}'.")
        return
    
    # Set center if provided
    center = None
    if args.center_x is not None and args.center_y is not None:
        center = (args.center_x, args.center_y)
    
    # Rotate the image
    rotated_image = rotate_image(image, args.angle, center, args.scale)
    
    # Save the rotated image
    cv2.imwrite(args.output_image, rotated_image)
    print(f"Rotated image saved as '{args.output_image}'")

if __name__ == "__main__":
    main()
