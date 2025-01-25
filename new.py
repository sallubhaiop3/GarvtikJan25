from textblob import TextBlob
import re
import eqi.py

# Define patterns and responses
pairs = [
    [r"hi|hello|hey", ["Hello! How can I help you today?"]],
    [r"what is your name?", ["I'm a chatbot, and my name is Chatty!"]],
    [r"how are you?", ["I'm just a program, but I'm functioning well! How about you?"]],
    [r"quit", ["Goodbye! Have a greabdt day!"]],
]

emotion_responses = {
    "happy": "I'm so glad to hear that! 😊 What's making you happy?",
    "positive": "That's great to hear! How can I assist you further?",
    "neutral": "Got it. Let me know how I can help you.",
    "negative": "I'm sorry to hear that. Is there something I can do to help?",
    "sad": "I'm here for you. It's okay to feel this way sometimes."
}

# Emotion detection function
def detect_emotion(user_input):
    analysis = TextBlob(user_input)
    sentiment = analysis.sentiment.polarity

    if sentiment > 0.5:
        return "happy"
    elif sentiment > 0:
        return "positive"
    elif sentiment == 0:
        return "neutral"
    elif sentiment > -0.5:
        return "negative"
    else:
        return "sad"

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
            emotion = detect_emotion(user_input)
            
            # Step 4: Respond Based on Emotion
            response = emotion_responses.get(emotion, "I'm not sure how to respond to that.")
            print(f"Chatbot: {response}")

# Start the chatbot
chatbot()

