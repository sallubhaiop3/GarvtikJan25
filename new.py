from transformers import pipeline, GPTNeoForCausalLM, AutoTokenizer
from deepface import DeepFace
import cv2
import re

# Load the emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

# Define patterns and responses
pairs = [
    [r"hi|hello|hey", ["Hello! How can I help you today?"]],
    [r"what is your name?", ["I'm a chatbot, and my name is Chatty!"]],
    [r"how are you?", ["I'm just a program, but I'm functioning well! How about you?"]],
    [r"quit", ["Goodbye! Have a great day!"]],
]   

emotion_responses = {
    "joy": "I'm so glad to hear that! 😊 What's making you happy?",
    "positive": "That's great to hear! How can I assist you further?",
    "neutral": "Got it. Let me know how I can help you.",
    "anger": "I'm sorry you're feeling angry. Would you like to talk about it?",
    "sadness": "I'm here for you. It's okay to feel this way sometimes.",
    "fear": "I'm sorry you're feeling fearful. Is there something I can do to help?",
    "surprise": "That sounds exciting! What happened?",
    "disgust": "I'm sorry you're feeling this way. Let me know if I can help."
}

# Load GPT-Neo model and tokenizer from Hugging Face
model_name = "EleutherAI/gpt-neo-1.3B"  # You can switch to other variants such as gpt-neo-2.7B for better performance
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token to avoid padding errors
tokenizer.pad_token = tokenizer.eos_token

# Text-based emotion detection function
def detect_emotion_text(user_input):
    try:
        result = emotion_classifier(user_input)
        # Extract the top emotion and confidence score
        top_emotion = result[0]['label']
        confidence = result[0]['score']
        return top_emotion, confidence
    except Exception as e:
        return "neutral", 0.0  # Default fallback

# Face-based emotion detection function
def detect_emotion_face():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to capture your emotion and exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera.")
            break

        cv2.imshow("Emotion Detection - Press 'q' to capture", frame)

        # Press 'q' to capture the frame and analyze
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("temp.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    try:
        result = DeepFace.analyze(img_path="temp.jpg", actions=["emotion"], enforce_detection=True)
        if isinstance(result, dict) and "dominant_emotion" in result:
            return result["dominant_emotion"]
        else:
            return "neutral"
    except Exception as e:
        print("Error detecting emotion:", str(e))
        return "neutral"

# GPT-Neo API call
def call_gptneo_api(user_input):
    try:
        # Tokenize input and generate a response with attention mask
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)

        # Decode the output and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print("Error with GPT-Neo API:", str(e))
        return "I'm sorry, I couldn't process your request right now."

# Chatbot function
def chatbot():
    print("Chatbot is running with Emotional Intelligence and GPT-Neo! Type 'quit' to exit.")
    print("If you'd like to use facial emotion detection, type 'detect face'.")
    
    while True:
        # Step 1: User Input
        user_input = input("You: ").lower()
        
        if user_input == "quit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        elif user_input == "detect face":
            print("Chatbot: Activating camera for facial emotion detection...")
            emotion = detect_emotion_face()
            print(f"Detected Emotion: {emotion}")
            response = emotion_responses.get(emotion, "I'm not sure how to respond to that.")
            print(f"Chatbot: {response}")
            continue
        
        # Step 2: Match Predefined Patterns
        for pattern, responses in pairs:
            if re.search(pattern, user_input):
                print(f"Chatbot: {responses[0]}")
                break
        else:
            # Step 3: Use GPT-Neo for Custom Responses
            gptneo_response = call_gptneo_api(user_input)
            print(f"GPT-Neo: {gptneo_response}")

# Start the chatbot
chatbot()
