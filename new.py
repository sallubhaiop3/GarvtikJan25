from transformers import pipeline
from deepface import DeepFace
import cv2
import re
from llama_cpp import Llama
import os
import time
from huggingface_hub import hf_hub_download
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# pairs = [
#     [r"hi|hello|hey", ["Hello! How can I help you today?"]],
#     [r"what is your name?", ["I'm a chatbot, and my name is Chatty!"]],
#     [r"how are you?", ["I'm just a program, but I'm functioning well! How about you?"]],
#     [r"quit", ["Goodbye! Have a great day!"]],
# ]   

# emotion_responses = {
#     "joy": "I'm so glad to hear that! 😊 What's making you happy?",
#     "positive": "That's great to hear! How can I assist you further?",
#     "neutral": "Got it. Let me know how I can help you.",
#     "anger": "I'm sorry you're feeling angry. Would you like to talk about it?",
#     "sadness": "I'm here for you. It's okay to feel this way sometimes.",
#     "fear": "I'm sorry you're feeling fearful. Is there something I can do to help?",
#     "surprise": "That sounds exciting! What happened?",
#     "disgust": "I'm sorry you're feeling this way. Let me know if I can help."
# }
# Check if model exists
def download_small_model():
    print("Downloading smaller model...")
    model_path = hf_hub_download(
        repo_id="TheBloke/phi-2-GGUF",  # Changed to Phi-2 model
        filename="phi-2.Q4_K_M.gguf",    # Different filename
        local_dir="./models"
    )
    return model_path

# Check if model exists
model_path = os.path.join("models", "phi-2.Q4_K_M.gguf")
if not os.path.exists(model_path):
    print("Downloading smaller model...")
    model_path = download_small_model()

# Initialize LLaMA model globally
llm = None

def initialize_model():
    global llm
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=256,
            n_threads=1,
            n_batch=1,
            verbose=False
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        exit(1)

def cleanup():
    global llm
    if llm is not None:
        del llm
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)

def get_llama_response(user_input, emotion=None):
    try:
        if emotion:
            prompt = f"As a helpful assistant, respond to a user showing {emotion} emotion. Be brief and empathetic."
        else:
            prompt = f"User: {user_input}\nAssistant:"
            
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=20,
            temperature=0.1,
            stop=["User:", "\n"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error in get_llama_response: {str(e)}")
        return "I'm here to help. How are you feeling?"

def chatbot():
    try:
        print("Chatbot is running! Type 'quit' to exit.")
        print("Type 'detect face' to use facial emotion detection.")
        
        while True:
            user_input = input("You: ").lower().strip()
            
            if user_input == "quit":
                print("Chatbot: Goodbye! Have a great day!")
                break
                
            elif user_input == "detect face":
                print("Chatbot: Activating camera...")
                emotion = detect_emotion_face()
                print(f"Detected Emotion: {emotion}")
                response = get_llama_response("", emotion)
                print(f"Chatbot: {response}")
                continue
            
            response = get_llama_response(user_input)
            print(f"Chatbot: {response}")
            
    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
    finally:
        cleanup()

def detect_emotion_face():
    camera = None
    try:
        print("Opening camera...")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Error: Could not open camera")
            return "neutral"

        print("Camera preview starting... Press 'q' to capture or wait 5 seconds")
        start_time = time.time()
        emotions_detected = []
        
        # Show preview and collect emotions
        while (time.time() - start_time) < 5:  # 5 second preview
            ret, frame = camera.read()
            if ret:
                cv2.imshow('Camera Preview', frame)
                
                # Analyze emotion every second
                if len(emotions_detected) < 5:  # Collect up to 5 samples
                    try:
                        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        emotions_detected.append(result[0]['emotion'])
                    except:
                        pass
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to read frame")
                break
        
        # Get the most confident emotion
        if emotions_detected:
            # Average the emotion scores
            emotion_scores = {}
            for emotion_dict in emotions_detected:
                for emotion, score in emotion_dict.items():
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score
            
            # Get the emotion with highest average score
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            print(f"Detected emotions: {emotion_scores}")
            print(f"Final emotion: {dominant_emotion}")
            return dominant_emotion
        else:
            print("No emotions detected, returning neutral")
            return "neutral"

    except Exception as e:
        print(f"Camera error: {str(e)}")
        return "neutral"
        
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        # Force close windows
        for i in range(4):
            cv2.waitKey(1)

if __name__ == "__main__":
    initialize_model()
    try:
        chatbot()
    finally:
        cleanup()