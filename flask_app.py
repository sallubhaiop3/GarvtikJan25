from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to process user messages
def get_response(user_message):
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a program, but I'm functioning perfectly!",
        "bye": "Goodbye! Have a great day!",
    }
    return responses.get(user_message.lower(), "I'm sorry, I didn't understand that.")
@app.route("/chat", methods=["GET"])
def chat():
    try:
        # Log the incoming request body to check what is being received
        print("Received request:", request.data)  # This will log the raw data
 
        # Check if the request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
        
        user_message = request.json.get("message")
        if user_message:
            response = f"Received your message: {user_message}"
            return jsonify({"response": response})
        else:
            return jsonify({"response": "No message found!"}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 400
# Home route
@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask app is running!"

if __name__ == "__main__":
    app.run(debug=True)