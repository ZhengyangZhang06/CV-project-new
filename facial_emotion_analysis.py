import argparse
import os
import torch
import cv2

# Project modules
from models.resnet import ResNet18
from utils.emotion_recognition import EMOTIONS, EMOTION_COLORS
from utils.video_processing import (
    process_video_with_onnx, 
    merge_audio
)

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial Emotion Analysis on Video using ONNX face detection')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video (optional)')
    parser.add_argument('--sample_rate', type=int, default=15, help='Frames per second to process (default: 15)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display during processing')
    parser.add_argument('--face_model', help='Path to the ONNX face detection model file (optional)')
    parser.add_argument('--no_audio', action='store_true', help='Do not merge audio from original video')
    args = parser.parse_args()
    
    # Load the emotion recognition model
    model_path = 'models/fer2013_resnet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded emotion recognition model from {model_path}")
    
    # Get video path if not provided as argument
    video_path = args.video
    if not video_path:
        video_path = input("Enter the path to your video file: ")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    # Get face model path if not provided as argument
    face_model_path = args.face_model
    if not face_model_path and args.video is None:  # interactive mode
        model_option = input("Do you want to specify a path to the face detection model? (y/n): ").strip().lower()
        if model_option == 'y':
            face_model_path = input("Enter the path to the ONNX face detection model: ").strip()
    
    # Get output path if not provided
    output_path = args.output
    if not output_path:
        save_option = input("Do you want to save the processed video? (y/n): ").strip().lower()
        if save_option == 'y':
            output_path = input("Enter the output path (or press Enter for default): ").strip()
            if not output_path:
                output_path = 'output_' + os.path.basename(video_path)
    
    # Ask if user wants to see the processing (only in interactive mode)
    display = not args.no_display
    if args.video is None:  # If we're in interactive mode
        display_option = input("Do you want to display video while processing? (y/n): ").strip().lower()
        display = display_option == 'y'
    
    # Process the video with ONNX face detection
    process_video_with_onnx(
        video_path, 
        model, 
        DEVICE,
        output_path=output_path,
        sample_rate=args.sample_rate,
        display=display,
        face_model_path=face_model_path
    )
    
    # Merge audio if output was generated and audio merging is not disabled
    if output_path and not args.no_audio:
        # Ask if user wants to add audio (only in interactive mode)
        add_audio = True
        if args.video is None:  # If we're in interactive mode
            audio_option = input("Do you want to merge audio from the original video? (y/n): ").strip().lower()
            add_audio = audio_option == 'y'
        
        if add_audio:
            print("Merging audio from original video...")
            # The merge_audio function will now overwrite the processed video if successful
            final_path = merge_audio(video_path, output_path)
            
            if final_path:
                print(f"Final video with audio saved to: {final_path}")
            else:
                print("Failed to merge audio. The processed video will not have audio.")
    
    print("Video processing complete!")

if __name__ == "__main__":
    main()
