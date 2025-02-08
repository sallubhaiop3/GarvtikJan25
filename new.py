import os
import cv2
import time
import random
import warnings
import numpy as np
import speech_recognition as sr
import pyttsx3
from deepface import DeepFace
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import multiprocessing

# Suppress warnings
warnings.filterwarnings('ignore')
multiprocessing.set_start_method('spawn', force=True)

class EmotionalChatbot:
    def __init__(self):
        self.initialize_all_components()
        
    def initialize_all_components(self):
        """Initialize all components of the chatbot."""
        print("Initializing chatbot components...")
        self.initialize_models()
        self.initialize_speech_components()
        self.initialize_emotion_responses()
        self.conversation_history = []
        print("Initialization complete!")

    def initialize_models(self):
        """Initialize all AI models."""
        # Download and setup LLaMA model
        self.model_path = os.path.join("models", "phi-2.Q4_K_M.gguf")
        if not os.path.exists(self.model_path):
            print("Downloading language model...")
            self.model_path = self.download_llama_model()

        # Initialize LLaMA
        print("Loading language model...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=512,
            n_threads=4,
            n_batch=8
        )

        # Initialize text emotion analyzer
        print("Loading emotion analyzer...")
        self.text_emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

        # Initialize BlenderBot
        print("Loading conversation model...")
        model_name = "facebook/blenderbot-400M-distill"
        self.chat_tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.chat_model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def initialize_speech_components(self):
        """Initialize speech recognition and synthesis."""
        print("Setting up speech components...")
        # Text to Speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)    # Speaking rate
        self.tts_engine.setProperty('volume', 0.9)  # Volume level
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Adjust for ambient noise
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def initialize_emotion_responses(self):
        """Initialize emotional response templates."""
        self.emotion_responses = {
            "happy": [
                "Your happiness is radiating! What wonderful thing happened?",
                "That smile is contagious! Share your joy with me!",
                "I can see you're in great spirits! What's the good news?"
            ],
            "sad": [
                "I notice you're feeling down. Would you like to talk about it?",
                "It's okay to feel sad. I'm here to listen if you need someone.",
                "Sometimes sharing our sorrows makes them lighter. What's troubling you?"
            ],
            "angry": [
                "I can see you're frustrated. Want to talk about what's bothering you?",
                "Your anger is valid. Let's discuss what's causing these feelings.",
                "Sometimes anger helps us identify what's wrong. What's upsetting you?"
            ],
            "fear": [
                "I sense you're feeling anxious. What's causing your concern?",
                "It's natural to feel scared sometimes. Want to share what's worrying you?",
                "I'm here to support you through this. What's making you fearful?"
            ],
            "surprise": [
                "You look amazed! What unexpected thing happened?",
                "That's quite a surprised expression! What caught you off guard?",
                "Something seems to have really shocked you! Want to share?"
            ],
            "disgust": [
                "Something seems to be really bothering you. Want to talk about it?",
                "I can see you're disturbed by something. What's wrong?",
                "Your expression shows strong disapproval. What's troubling you?"
            ],
            "neutral": [
                "How are you feeling right now?",
                "What's on your mind?",
                "I'm here to chat about whatever you'd like."
            ]
        }

    def download_llama_model(self):
        """Download the LLaMA model."""
        return hf_hub_download(
            repo_id="TheBloke/phi-2-GGUF",
            filename="phi-2.Q4_K_M.gguf",
            local_dir="./models"
        )

    def detect_facial_emotion(self):
        """Detect emotion from facial expression."""
        camera = None
        try:
            print("\nStarting facial emotion detection...")
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                print("Error: Cannot access camera")
                return "neutral"

            print("Camera active. Press 'q' to capture or wait 5 seconds...")
            start_time = time.time()
            emotions_detected = []
            
            while (time.time() - start_time) < 5:
                ret, frame = camera.read()
                if ret:
                    # Display frame
                    cv2.imshow('Emotion Detection', frame)
                    
                    # Analyze emotion
                    if len(emotions_detected) < 5:
                        try:
                            result = DeepFace.analyze(
                                frame, 
                                actions=['emotion'],
                                enforce_detection=False
                            )
                            emotions_detected.append(result[0]['emotion'])
                        except Exception as e:
                            pass
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Camera read error")
                    break

            # Process detected emotions
            if emotions_detected:
                # Aggregate emotion scores
                emotion_scores = {}
                for emotion_dict in emotions_detected:
                    for emotion, score in emotion_dict.items():
                        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

                # Get dominant emotion
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                
                # Generate immediate emotional response
                response = random.choice(self.emotion_responses[dominant_emotion.lower()])
                print(f"\nDetected Emotion: {dominant_emotion}")
                print(f"Bot: {response}")
                self.speak(response)
                
                return dominant_emotion.lower()
            
            return "neutral"

        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return "neutral"
            
        finally:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)

    def generate_emotional_response(self, user_input, emotion):
        """Generate an emotionally appropriate response."""
        try:
            # Create emotion-aware prompt
            prompt = f"""As an empathetic AI assistant talking to someone feeling {emotion}, 
            generate a caring response to: "{user_input}". Be supportive and understanding 
            while maintaining a natural conversation flow."""
            
            # Generate response
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["User:", "\n"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Response generation error: {str(e)}")
            return random.choice(self.emotion_responses[emotion])

    def speak(self, text):
        """Convert text to speech."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {str(e)}")

    def listen(self):
        """Listen for voice input."""
        with self.mic as source:
            print("\nListening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError:
                print("Speech recognition service unavailable.")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {str(e)}")
                return None

    def chat(self):
        """Main chat loop."""
        print("\nWelcome to Emotional Chatbot!")
        print("\nCommands:")
        print("- 'voice': Toggle voice mode")
        print("- 'face': Detect facial emotion")
        print("- 'exit': End chat")
        
        voice_mode = False
        current_emotion = "neutral"
        
        while True:
            try:
                # Get user input
                if voice_mode:
                    user_input = self.listen()
                    if not user_input:
                        continue
                else:
                    user_input = input("\nYou: ").strip()

                # Process commands
                if user_input.lower() == "exit":
                    farewell = "Goodbye! Take care!"
                    print(f"Bot: {farewell}")
                    self.speak(farewell)
                    break
                    
                elif user_input.lower() == "voice":
                    voice_mode = not voice_mode
                    status = "activated" if voice_mode else "deactivated"
                    message = f"Voice mode {status}!"
                    print(f"Bot: {message}")
                    self.speak(message)
                    continue
                    
                elif user_input.lower() == "face":
                    current_emotion = self.detect_facial_emotion()
                    continue

                # Generate response
                response = self.generate_emotional_response(user_input, current_emotion)
                print(f"Bot: {response}")
                self.speak(response)

                # Update conversation history
                self.conversation_history.append(f"User ({current_emotion}): {user_input}")
                self.conversation_history.append(f"Bot: {response}")

            except Exception as e:
                print(f"Error in chat loop: {str(e)}")
                continue

    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        if hasattr(self, 'llm'):
            del self.llm

def main():
    """Main function to run the chatbot."""
    print("Starting Emotional Chatbot System...")
    
    # Create required directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize and run chatbot
    chatbot = EmotionalChatbot()
    try:
        chatbot.chat()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        chatbot.cleanup()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()