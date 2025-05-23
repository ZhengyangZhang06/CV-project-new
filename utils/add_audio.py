def add_audio_to_video(original_video, processed_video, output_video=None):
    """
    Add audio from original video to the processed video.
    
    Args:
        original_video (str): Path to the original video with audio
        processed_video (str): Path to the processed video without audio
        output_video (str, optional): Path to save the output video with audio.
                                     If None, will be automatically generated.
    
    Returns:
        str: Path to the output video with audio
    """
    import os
    import sys
    import shutil
    
    # Check if files exist first
    if not os.path.exists(original_video):
        print(f"Error: Original video file '{original_video}' not found.")
        return None
        
    if not os.path.exists(processed_video):
        print(f"Error: Processed video file '{processed_video}' not found.")
        return None
    
    # Create output path if not provided
    if output_video is None:
        base_name = os.path.splitext(processed_video)[0]
        ext = os.path.splitext(processed_video)[1]
        output_video = f"{base_name}_with_audio{ext}"
    
    print(f"Merging audio from '{original_video}' into '{processed_video}'...")
    
    # First try using MoviePy
    try:
        # Try to import moviepy
        try:
            from moviepy import VideoFileClip
        except ImportError:
            print("MoviePy package not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
            from moviepy import VideoFileClip
            
        # Load the processed video (without audio)
        video_clip = VideoFileClip(processed_video)
        
        # Load the original video (with audio)
        original_clip = VideoFileClip(original_video)
        
        # Check if original video has audio
        if original_clip.audio is None:
            print("Original video does not have an audio track.")
            video_clip.close()
            original_clip.close()
            return None
        
        # Use with_audio method (compatible with VS Code's player)
        final_clip = video_clip.with_audio(original_clip.audio)
        
        # Write the result to a file with settings compatible with VS Code
        final_clip.write_videofile(output_video)
        
        # Close the clips to free resources
        try:
            video_clip.close()
            original_clip.close()
            final_clip.close()
        except Exception as close_error:
            print(f"Warning when closing clips: {close_error}")
            
        print(f"Audio successfully merged. Output saved to: {output_video}")
        
        # Overwrite the original processed video with the one containing audio
        try:
            print(f"Replacing original processed video with the version containing audio...")
            shutil.move(output_video, processed_video)
            print(f"Successfully replaced {processed_video} with the version containing audio")
            return processed_video
        except Exception as move_error:
            print(f"Failed to replace original video: {move_error}")
            return output_video
        
    except Exception as e:
        print(f"MoviePy error: {str(e)}")
        
        # Close any open clips
        try:
            if 'video_clip' in locals(): video_clip.close()
            if 'original_clip' in locals(): original_clip.close()
            if 'final_clip' in locals(): final_clip.close()
        except:
            pass
            
        # Try the direct FFmpeg approach as fallback
        try:
            print("Trying with FFmpeg as fallback...")
            import subprocess
            
            # Using a temporary output path for ffmpeg
            temp_output = output_video
            
            cmd = [
                "ffmpeg",
                "-i", processed_video,
                "-i", original_video,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                "-pix_fmt", "yuv420p",
                "-y",
                temp_output
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Successfully merged audio using ffmpeg. Output saved to: {temp_output}")
            
            # Overwrite the original processed video with the one containing audio
            try:
                print(f"Replacing original processed video with the version containing audio...")
                shutil.move(temp_output, processed_video)
                print(f"Successfully replaced {processed_video} with the version containing audio")
                return processed_video
            except Exception as move_error:
                print(f"Failed to replace original video: {move_error}")
                return temp_output
                
        except Exception as e2:
            print(f"All audio merging methods failed: {str(e2)}")
            return None

def merge_audio(original_video, processed_video, output_video=None):
    """
    Alias for add_audio_to_video function. 
    Add audio from original video to the processed video.
    
    Args:
        original_video (str): Path to the original video with audio
        processed_video (str): Path to the processed video without audio
        output_video (str, optional): Path to save the output video with audio.
                                     If None, will be automatically generated.
    
    Returns:
        str: Path to the output video with audio
    """
    return add_audio_to_video(original_video, processed_video, output_video)
