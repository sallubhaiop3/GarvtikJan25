from transformers import pipeline
import re

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

# Load the emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

# Emotion detection function
def detect_emotion(user_input):
    try:
        result = emotion_classifier(user_input)
        # Extract the top emotion and confidence score
        top_emotion = result[0]['label']
        confidence = result[0]['score']
        return top_emotion, confidence
    except Exception as e:
        
        return "neutral", 0.0  # Default fallback

# Chatbot function
def chatbot():
    print("Chatbot is running with Emotional Intelligence! Type 'quit' to exit.")
    
    while True:
        # Step 1: User Input
        user_input = input("You: ").lower()
        
        if user_input == "quit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Step 2: Match Predefined Patterns
        for pattern, responses in pairs:
            if re.search(pattern, user_input):
                print(f"Chatbot: {responses[0]}")
                break
        else:
            # Step 3: Detect Emotion
            emotion, confidence = detect_emotion(user_input)
        
            
            # Step 4: Respond Based on Emotion
            response = emotion_responses.get(emotion, "I'm not sure how to respond to that.")
            print(f"Chatbot: {response}")   

# Start the chatbot
chatbot()
