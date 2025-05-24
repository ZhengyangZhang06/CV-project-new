import os
import random
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

def run_emotion_test(image_dir="emotion_images", num_questions=10):
    """
    Show random emotion images and test user's ability to identify emotions
    
    Args:
        image_dir: Directory containing emotion subdirectories with images
        num_questions: Number of test questions to ask
    """
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found!")
        return
    
    # Track correct answers
    correct_count = 0
    answers = []
    
    print("\n===== EMOTION RECOGNITION TEST =====")
    print("I'll show you random facial expressions and you'll guess the emotion.")
    print("Input the number for your answer:")
    for idx, emotion in EMOTIONS.items():
        print(f"  {idx+1} = {emotion.capitalize()}")
    print("\nLet's begin!")
    
    # Create list of all image paths grouped by emotion
    emotion_images = {}
    for emotion_id, emotion_name in EMOTIONS.items():
        emotion_dir = os.path.join(image_dir, emotion_name)
        if os.path.exists(emotion_dir):
            image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg'))]
            if image_files:
                emotion_images[emotion_id] = image_files
    
    if not emotion_images:
        print("Error: No emotion images found!")
        return
        
    # Select random images
    questions = []
    available_emotions = list(emotion_images.keys())
    
    for _ in range(num_questions):
        emotion_id = random.choice(available_emotions)
        image_path = random.choice(emotion_images[emotion_id])
        questions.append((emotion_id, image_path))
    
    # Ask each question
    for q_num, (correct_id, image_path) in enumerate(questions, 1):
        # Display image
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Question {q_num}/{num_questions}: What emotion is this?")
        plt.axis('off')
        plt.show(block=False)
        
        # Get user's answer
        while True:
            try:
                user_answer = int(input(f"\nQuestion {q_num}: Enter number (1-7): "))
                if 1 <= user_answer <= 7:
                    break
                else:
                    print("Please enter a number between 1 and 7.")
            except ValueError:
                print("Invalid input. Enter a number between 1 and 7.")
        
        # Convert to 0-based index
        user_answer -= 1
        correct_emotion = EMOTIONS[correct_id]
        
        # Check answer
        is_correct = (user_answer == correct_id)
        if is_correct:
            correct_count += 1
            result = "Correct! ✓"
        else:
            result = f"Incorrect. ✗ (The correct answer was: {correct_emotion})"
        
        answers.append((user_answer, correct_id, is_correct))
        print(result)
        plt.close()
    
    # Calculate accuracy
    accuracy = (correct_count / num_questions) * 100
    
    # Display results
    print("\n===== TEST RESULTS =====")
    print(f"You got {correct_count} out of {num_questions} correct.")
    print(f"Your accuracy: {accuracy:.1f}%")
    
    # Show breakdown by emotion
    print("\nBreakdown by emotion:")
    emotion_stats = {emotion_id: {"total": 0, "correct": 0} for emotion_id in EMOTIONS}
    
    for user_ans, correct_id, is_correct in answers:
        emotion_stats[correct_id]["total"] += 1
        if is_correct:
            emotion_stats[correct_id]["correct"] += 1
    
    for emotion_id, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"{EMOTIONS[emotion_id].capitalize()}: {stats['correct']}/{stats['total']} ({emotion_accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    print("Welcome to the Emotion Recognition Test!")
    
    # Get number of questions
    while True:
        try:
            num_questions = input("How many questions would you like? (default: 10): ")
            num_questions = 10 if not num_questions else int(num_questions)
            if num_questions > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get image directory
    image_dir = input("Enter image directory path (default: emotion_images): ") or "emotion_images"
    
    # Run the test
    run_emotion_test(image_dir, num_questions)