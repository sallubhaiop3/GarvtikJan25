from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
from deepface import DeepFace
from llama_cpp import Llama
import os
import socket
import traceback  # Add this for detailed error tracking
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
import concurrent.futures
import atexit

app = Flask(__name__)

# Initialize LLaMA model globally
llm = None

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def initialize_model():
    global llm
    try:
        # Initialize model with proper error handling
        llm = Llama(
            model_path="models/phi-2.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=4,
            n_batch=512
        )
        # Register cleanup function
        atexit.register(cleanup_model)
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

def cleanup_model():
    global llm
    try:
        if llm is not None:
            del llm
            llm = None
    except Exception as e:
        print(f"Error cleaning up model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global llm
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if llm is None:
            return jsonify({'response': 'Model not initialized properly'}), 500
            
        # Log received message
        print(f"Received message: {message}")
        
        # Generate response with proper error handling
        response = llm.create_completion(
            message,
            max_tokens=128,
            stop=["User:", "\n"],
            echo=False
        )
        
        return jsonify({'response': response['choices'][0]['text'].strip()})
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'response': "I'm having trouble processing your message right now."}), 500

def save_emotion_capture(image_data, emotion, confidence):
    """Save emotion capture to database and file system"""
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'emotion_{timestamp}.jpg'
        filepath = f'static/uploads/{filename}'
        
        # Save image file
        # Remove the data URL prefix to get just the base64 data
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Save to database
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO emotion_captures 
            (image_path, emotion, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (filepath, emotion, confidence, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error saving emotion capture: {str(e)}")
        return False

def detect_emotion_in_image(image_data):
    try:
        # Convert base64 to image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        
        return emotion, confidence
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return "unknown", 0.0

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        image_data = data['image']  # This is base64 encoded image
        
        # Detect emotion using your model
        emotion, confidence = detect_emotion_in_image(image_data)
        
        # Save the capture
        if save_emotion_capture(image_data, emotion, confidence):
            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'message': 'Emotion captured and saved successfully'
            })
        else:
            return jsonify({
                'error': 'Failed to save emotion capture'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def view_history():
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute('''
            SELECT id, image_path, emotion, confidence, timestamp 
            FROM emotion_captures 
            ORDER BY timestamp DESC
        ''')
        history = c.fetchall()
        conn.close()
        return render_template('history.html', history=history)
    except Exception as e:
        print(f"Error accessing history: {e}")
        return render_template('history.html', history=[])

@app.route('/image/<int:id>')
def view_image(id):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM emotion_captures WHERE id = ?', (id,))
    capture = c.fetchone()
    conn.close()
    if capture:
        return render_template('image.html', capture=capture)
    return "Image not found", 404

@app.route('/delete-photo/<int:id>', methods=['POST'])
def delete_photo(id):
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        
        # Get image path before deletion
        c.execute('SELECT image_path FROM emotion_captures WHERE id = ?', (id,))
        result = c.fetchone()
        
        if result:
            image_path = result[0]
            
            # Delete from database
            c.execute('DELETE FROM emotion_captures WHERE id = ?', (id,))
            conn.commit()
            
            # Delete file from filesystem
            if os.path.exists(image_path):
                os.remove(image_path)
                
            return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/view-photo/<int:id>')
def view_photo(id):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM emotion_captures WHERE id = ?', (id,))
    photo = c.fetchone()
    conn.close()
    
    if photo:
        return render_template('view-photo.html', photo=photo)
    return "Photo not found", 404

def find_free_port():
    """Find a free port on the system"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def init_db():
    """Initialize the database and create required tables"""
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        
        # Create table for emotion captures
        c.execute('''
            CREATE TABLE IF NOT EXISTS emotion_captures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        
    finally:
        conn.close()

# Enable debugging and auto-reload
if __name__ == '__main__':
    # Development settings
    app.config['DEBUG'] = True
    
    # Generate self-signed certificates if they don't exist
    if not (os.path.exists('cert.pem') and os.path.exists('key.pem')):
        from OpenSSL import crypto
        # Generate key
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # Generate certificate
        cert = crypto.X509()
        cert.get_subject().CN = 'localhost'
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # Save certificate and private key
        with open('cert.pem', 'wb') as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open('key.pem', 'wb') as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    # Run with HTTPS
    app.run(
        debug=True,
        host='0.0.0.0',
        port=4000,
    )
# Your existing routes and functions below 