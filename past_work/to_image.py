import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define emotion labels mapping (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
EMOTIONS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

def csv_to_images(csv_file, output_dir='emotion_images', image_size=48):
    """
    Convert CSV file with emotion and pixels to images
    
    Args:
        csv_file: Path to CSV file containing emotions and pixels
        output_dir: Directory to save images
        image_size: Size of the square images (default: 48x48 pixels)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create subdirectories for each emotion
    for emotion_id, emotion_name in EMOTIONS.items():
        emotion_dir = os.path.join(output_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required_cols = ['emotion', 'pixels']
    if not all(col in df.columns for col in required_cols):
        # Try to adapt to different CSV formats
        if 'emotion' not in df.columns and len(df.columns) >= 2:
            # Assume first column is emotion, second is pixels
            df.columns = ['emotion'] + list(df.columns[1:])
            df = df.rename(columns={df.columns[1]: 'pixels'})
            print("Adapting to CSV format: first column as emotion, second as pixels")
        else:
            print(f"Error: CSV must contain columns {required_cols}")
            return
    
    total_images = len(df)
    print(f"Found {total_images} images in the CSV file")
    
    # Process each row and save as image
    for idx, row in df.iterrows():
        try:
            # Get emotion ID and name
            emotion_id = int(row['emotion'])
            emotion_name = EMOTIONS.get(emotion_id, 'unknown')
            
            # Convert pixel string to numpy array and reshape
            pixel_str = row['pixels']
            pixels = np.array([int(p) for p in pixel_str.split()], dtype=np.uint8)
            
            # Check if we have the right number of pixels
            expected_pixels = image_size * image_size
            if len(pixels) != expected_pixels:
                print(f"Warning: Row {idx} has {len(pixels)} pixels, expected {expected_pixels}")
                continue
                
            # Reshape to square image
            image = pixels.reshape(image_size, image_size)
            
            # Create file name: emotion_rowindex.png
            file_name = f"{emotion_name}_{idx}.png"
            file_path = os.path.join(output_dir, emotion_name, file_name)
            
            # Save image
            img = Image.fromarray(image)
            img.save(file_path)
            
            # Print progress
            if (idx + 1) % 1000 == 0 or idx == 0 or idx == total_images - 1:
                print(f"Progress: {idx + 1}/{total_images} images processed ({(idx + 1) * 100 / total_images:.1f}%)")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    print(f"Conversion complete. Images saved to {output_dir}")

def display_sample_images(output_dir, samples_per_emotion=3):
    """Display sample images from each emotion category"""
    plt.figure(figsize=(15, 10))
    
    for i, emotion in enumerate(EMOTIONS.values()):
        emotion_dir = os.path.join(output_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        image_files = os.listdir(emotion_dir)[:samples_per_emotion]
        
        for j, img_file in enumerate(image_files):
            if j >= samples_per_emotion:
                break
                
            img_path = os.path.join(emotion_dir, img_file)
            img = Image.open(img_path)
            
            # Plot in a grid
            plt.subplot(len(EMOTIONS), samples_per_emotion, i*samples_per_emotion + j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{emotion}" if j == 0 else "")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_emotions.png'))
    plt.show()

if __name__ == "__main__":
    # Define the path to your CSV file
    csv_file = "fer2013/train.csv"  # Change this to your CSV file path
    output_directory = "emotion_images"
    
    # Convert CSV to images
    csv_to_images(csv_file, output_directory)
    
    # Display sample images
    display_sample_images(output_directory)
    
    print("Process completed!")