from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
from deepface import DeepFace
from llama_cpp import Llama
import os
import socket
import traceback  # Add this for detailed error tracking

app = Flask(__name__)

# Initialize LLaMA model globally
llm = None

def initialize_model():
    global llm
    try:
        model_path = os.path.join("models", "phi-2.Q4_K_M.gguf")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        print(f"Loading model from: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,         # Reduced threads for stability
            n_batch=8,          # Reduced batch size
            verbose=True,       # Enable verbose mode for debugging
            chat_format="phi"   # Ensure correct chat format
        )
        
        # Test the model
        test_prompt = "Hello, test message."
        test_response = llm.create_completion(
            prompt=test_prompt,
            max_tokens=10,
            temperature=0.7,
            stop=["User:", "\n"],
            echo=False
        )
        print(f"Test response: {test_response}")
        
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if llm is None:
            print("Model not initialized")
            return jsonify({
                'error': 'Model not initialized',
                'response': 'The chatbot is still initializing. Please try again.'
            })
        
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please type a message first.'
            })
        
        print(f"Received message: {user_message}")
        
        # Improved prompt formatting
        prompt = f"""
Human: {user_message}
Assistant: I'll help you with that.
"""
        
        try:
            # Generate response with adjusted parameters
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=150,        # Increased token limit
                temperature=0.8,       # Slightly increased temperature
                top_p=0.95,           # Added top_p sampling
                repeat_penalty=1.1,    # Prevent repetition
                stop=["Human:", "\n\n"],
                echo=False
            )
            
            print(f"Model response: {response}")
            
            if response and 'choices' in response and response['choices']:
                bot_response = response['choices'][0]['text'].strip()
                
                # Validate response content
                if len(bot_response) > 5:  # Ensure response is meaningful
                    print(f"Valid response: {bot_response}")
                    return jsonify({'response': bot_response})
                else:
                    print("Response too short, generating fallback")
                    # Generate a fallback response based on the message type
                    if '?' in user_message:
                        fallback = "That's an interesting question. Could you provide more details?"
                    else:
                        fallback = "I understand what you're saying. Could you elaborate more?"
                    return jsonify({'response': fallback})
            else:
                print("Invalid response structure")
                return jsonify({
                    'error': 'Invalid response',
                    'response': 'Could you rephrase that? I want to make sure I understand correctly.'
                })
                
        except Exception as e:
            print(f"Model generation error: {str(e)}")
            return jsonify({
                'error': str(e),
                'response': 'I had trouble processing that. Could you try asking in a different way?'
            })
            
    except Exception as e:
        print(f"General chat error: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': "I'm having trouble understanding. Please try again."
        })

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        # Get image data
        image_data = request.json.get('image')
        if not image_data:
            print("No image data received")
            return jsonify({
                'error': 'No image data',
                'emotion': 'neutral',
                'response': 'No image received. Please try again.'
            })

        try:
            # Process image with better error handling
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode image")
                raise ValueError("Failed to decode image")

            # Save debug image (optional - uncomment to debug)
            # cv2.imwrite('debug_image.jpg', frame)
            
            # Resize image for faster processing
            max_size = 640
            height, width = frame.shape[:2]
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Attempt emotion detection
            print("Starting emotion detection...")
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                align=True,
                silent=True
            )
            print("Emotion detection completed")
            
            if not result or not isinstance(result, list):
                print("Invalid detection result")
                raise ValueError("Invalid detection result")

            # Extract emotions
            emotion = result[0]['dominant_emotion']
            emotions = result[0]['emotion']
            print(f"Detected emotion: {emotion}")
            print(f"All emotions: {emotions}")
            
            # Get top emotions
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Generate response
            if llm:
                response = llm.create_completion(
                    prompt=f"The user is feeling {emotion}. Give a very brief, empathetic response:",
                    max_tokens=30,
                    temperature=0.7,
                    stop=["User:", "\n"],
                    echo=False
                )
                bot_response = response['choices'][0]['text'].strip()
            else:
                bot_response = f"I see that you're feeling {emotion}."
            
            return jsonify({
                'emotion': emotion,
                'emotions': dict(sorted_emotions),
                'response': bot_response
            })
            
        except ValueError as ve:
            print(f"Value Error in image processing: {str(ve)}")
            return jsonify({
                'error': str(ve),
                'emotion': 'neutral',
                'response': 'Could not process the image. Please try again with better lighting.'
            })
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return jsonify({
                'error': str(e),
                'emotion': 'neutral',
                'response': 'Could not detect emotion clearly. Please ensure good lighting and face the camera directly.'
            })
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({
            'error': str(e),
            'emotion': 'neutral',
            'response': "Sorry, something went wrong. Please try again."
        })

def find_free_port():
    """Find a free port on the system"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    if initialize_model():
        try:
            port = find_free_port()
            print(f"Starting server on port: {port}")
            app.run(debug=True, host='0.0.0.0', port=port)
        except Exception as e:
            print(f"Error starting server: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print("Failed to initialize model. Please check the model file and settings.") 