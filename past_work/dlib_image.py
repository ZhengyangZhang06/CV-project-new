import cv2
import dlib
import numpy as np
import argparse
import os

def detect_faces(image_path, scale_factor=None, output_path=None, show_image=True):
    """
    Detect faces in an image using dlib and draw landmarks
    
    Args:
        image_path: Path to the input image
        scale_factor: Factor to resize image (for faster processing)
        output_path: Path to save the output image
        show_image: Whether to display the image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Calculate scale_factor if not provided
    if scale_factor is None:
        target_height = 480
        image_height = image.shape[0]
        scale_factor = min(1.0, target_height / image_height)  # Don't upscale small images
        if scale_factor > 0.8:  # Only resize if the image is significantly larger than 480p
            scale_factor = 1.0
    
    print(f"Using scale_factor: {scale_factor}")
    
    # Resize the image if scale_factor is not 1.0
    if scale_factor < 1.0:
        small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        small_image = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        print(f"Error: Landmark predictor file '{predictor_path}' not found.")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces
    faces = detector(gray)
    print(f"Found {len(faces)} faces in the image")
    
    # Create a copy of the original image to draw on
    result_image = image.copy()
    
    # Process each detected face
    for face in faces:
        # Convert dlib rectangle to OpenCV rectangle coordinates
        x1 = int(face.left() / scale_factor)
        y1 = int(face.top() / scale_factor)
        x2 = int(face.right() / scale_factor)
        y2 = int(face.bottom() / scale_factor)
        
        # Draw rectangle around the face
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Draw facial landmarks
        for i in range(68):
            landmark_x = int(landmarks.part(i).x / scale_factor)
            landmark_y = int(landmarks.part(i).y / scale_factor)
            cv2.circle(result_image, (landmark_x, landmark_y), 2, (255, 0, 0), -1)
            # Optionally draw landmark number
            # cv2.putText(result_image, str(i), (landmark_x, landmark_y), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
    # Show the result
    if show_image:
        cv2.imshow("Face Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the result if output_path is provided
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")
    
    return result_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection using dlib on a single image')
    parser.add_argument('--image', help='Path to the input image file', default='jg0.png')
    parser.add_argument('--output', help='Path to save the output image (optional)')
    parser.add_argument('--scale', type=float, help='Scale factor for image processing (optional)')
    parser.add_argument('--no_display', action='store_true', help='Do not display the result image')
    args = parser.parse_args()
    
    # Get image path
    image_path = args.image
    if not image_path:
        image_path = input("Enter the path to your image file: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Get output path if specified
    output_path = args.output
    
    # Process the image
    detect_faces(
        image_path,
        scale_factor=args.scale,
        output_path=output_path,
        show_image=not args.no_display
    )

if __name__ == "__main__":
    main()