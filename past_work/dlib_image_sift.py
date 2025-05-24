import cv2
import dlib
import numpy as np
import argparse
import os

def visualize_sift_matches(reference_image, target_image, kp1, kp2, good_matches):
    """
    Visualize SIFT keypoints and matches between reference and target images
    
    Args:
        reference_image: Reference image
        target_image: Target image
        kp1: Keypoints from reference image
        kp2: Keypoints from target image
        good_matches: Good matches between keypoints
        
    Returns:
        Tuple containing (keypoints_ref_image, keypoints_target_image, matches_image)
    """
    # Draw keypoints on both images
    img_kp1 = cv2.drawKeypoints(reference_image, kp1, None, color=(0, 255, 0), 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp2 = cv2.drawKeypoints(target_image, kp2, None, color=(0, 255, 0), 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Draw matches between images
    img_matches = cv2.drawMatches(reference_image, kp1, target_image, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Return all visualizations
    return (img_kp1, img_kp2, img_matches)

def align_face_with_sift(reference_image_path, target_image, visualize_matches=False):
    """
    Align a potentially distorted face image with a reference image using SIFT
    
    Args:
        reference_image_path: Path to the reference image
        target_image: The image to be aligned
        visualize_matches: Whether to visualize SIFT keypoints and matches
        
    Returns:
        Aligned image if successful, otherwise the original image
        Visualization images if visualize_matches is True
    """
    # Load the reference image
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Error: Could not load reference image from {reference_image_path}")
        return target_image, None if visualize_matches else target_image
    
    # Convert images to grayscale for SIFT
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    
    # FLANN parameters for fast matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Use FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    # Create visualization if requested
    visualizations = None
    if visualize_matches:
        visualizations = visualize_sift_matches(reference_image, target_image, kp1, kp2, good_matches)
    
    # Need at least 4 good matches to find homography
    if len(good_matches) >= 4:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Warp the image
            h, w = reference_image.shape[:2]
            aligned_image = cv2.warpPerspective(target_image, H, (w, h))
            print(f"Successfully aligned image using SIFT")
            return (aligned_image, visualizations) if visualize_matches else aligned_image
    
    print("Could not find enough matches to align the image")
    return (target_image, visualizations) if visualize_matches else target_image

def detect_faces(image_path, scale_factor=None, output_path=None, show_image=True, reference_image_path=None, visualize_sift=False):
    """
    Detect faces in an image using dlib and draw landmarks
    
    Args:
        image_path: Path to the input image
        scale_factor: Factor to resize image (for faster processing)
        output_path: Path to save the output image
        show_image: Whether to display the image
        reference_image_path: Path to reference image for SIFT alignment
        visualize_sift: Whether to visualize SIFT keypoints and matches
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Align the face if reference image is provided
    sift_visualizations = None
    if reference_image_path:
        print(f"Attempting to align using reference image: {reference_image_path}")
        if visualize_sift:
            image, sift_visualizations = align_face_with_sift(reference_image_path, image, visualize_matches=True)
        else:
            image = align_face_with_sift(reference_image_path, image)
    
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
    
    # Show SIFT visualization if available
    if visualize_sift and sift_visualizations is not None:
        ref_keypoints, target_keypoints, matches = sift_visualizations
        
        # Show keypoints first
        cv2.imshow("Reference Image Keypoints", ref_keypoints)
        cv2.waitKey(0)
        
        cv2.imshow("Target Image Keypoints", target_keypoints)
        cv2.waitKey(0)
        
        # Then show matches
        cv2.imshow("SIFT Matches Visualization", matches)
        cv2.waitKey(0)
        
        # Save SIFT visualizations if output path is specified
        if output_path:
            base_path = output_path.rsplit('.', 1)[0]
            ext = output_path.rsplit('.', 1)[1]
            
            cv2.imwrite(f"{base_path}_ref_keypoints.{ext}", ref_keypoints)
            cv2.imwrite(f"{base_path}_target_keypoints.{ext}", target_keypoints)
            cv2.imwrite(f"{base_path}_sift_matches.{ext}", matches)
            
            print(f"SIFT visualizations saved to {base_path}_*.{ext}")
    
    return result_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection using dlib on a single image')
    parser.add_argument('--image', help='Path to the input image file', default='aw_r180.png')
    parser.add_argument('--output', help='Path to save the output image (optional)')
    parser.add_argument('--scale', type=float, help='Scale factor for image processing (optional)')
    parser.add_argument('--no_display', action='store_true', help='Do not display the result image')
    parser.add_argument('--reference', help='Path to reference image for SIFT alignment', default='aw.png')
    parser.add_argument('--visualize_sift', action='store_true', help='Visualize SIFT keypoints and matches')
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
        show_image=not args.no_display,
        reference_image_path=args.reference,
        visualize_sift=args.visualize_sift
    )

if __name__ == "__main__":
    main()