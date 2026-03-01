import os
import time
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from mango import MangoEngine

# Determine absolute base path of the root 'mango' directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Define absolute paths based on your new architecture
frontend_dir = os.path.join(current_dir, 'frontend')
static_dir = os.path.join(frontend_dir, 'static')
output_dir = os.path.join(current_dir, 'output')

# Initialize Flask application
# Explicitly configure the template folder and the primary static asset folder (CSS/JS)
app = Flask(
    __name__, 
    template_folder=frontend_dir,
    static_folder=static_dir,
    static_url_path='/static'
)

# Ensure the output directory exists on startup so the engine doesn't crash on save
os.makedirs(output_dir, exist_ok=True)

# Initialize the inference engine
mango_engine = MangoEngine()

@app.route('/')
def home():
    """Renders the main application interface."""
    # Because we set template_folder to 'frontend', Flask knows to look inside it.
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Safely serves generated media files from the isolated output directory."""
    return send_from_directory(output_dir, filename)

@app.route('/generate_stream')
def generate_stream():
    """
    Handles Server-Sent Events (SSE) for image generation.
    Streams generation progress and yields the final image URL.
    """
    user_prompt = request.args.get('prompt', '').strip()
    
    if not user_prompt:
        return jsonify({'error': "Prompt parameter is required."}), 400

    # Generate a unique filename using a Unix timestamp
    filename = f"mango_art_{int(time.time())}.png"
    
    # Define the absolute OS path for the engine to save the file
    filepath = os.path.join(output_dir, filename)
    
    # Define the relative URL path that the browser will use to request the file
    url_path = f"output/{filename}"

    # Open the SSE stream and pass the prompt, save path, and URL path to the engine
    return Response(
        mango_engine.generate_stream(user_prompt, filepath, url_path), 
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    # '127.0.0.1' ensures the server is only accessible from this machine.
    # To allow other devices on your local network (like a phone or tablet) to access 
    # the interface, change '127.0.0.1' to '0.0.0.0'.
    # 
    # Once changed to '0.0.0.0', other devices can access the app by navigating to 
    # your computer's local IP address (e.g., http://192.168.1.XX:5000) in their browser.
    app.run(host='127.0.0.1', port=5000, debug=False)