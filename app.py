from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
import io
from tensorflow.keras.applications.vgg16 import preprocess_input
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.keras")
    CONFIDENCE_THRESHOLD = 0.7
    CLASS_LABELS = ["cat", "dog", "neither"]

# Create upload folder
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

def load_model_safe():
    try:
        model = load_model(Config.MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

# Load model globally
model = load_model_safe()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def process_image(image_data):
    img = load_img(image_data, target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Log the start of the request
    logger.info("=== Starting new classification request ===")
    logger.info(f"Request Method: {request.method}")
    logger.info(f"Content-Type: {request.headers.get('Content-Type', 'Not specified')}")
    logger.info(f"Files in request: {list(request.files.keys())}")

    if not model:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500

    # Modified file check
    if 'images' not in request.files:
        logger.warning("'images' not in request.files")
        logger.info(f"Available files: {list(request.files.keys())}")
        # Return success with a warning instead of an error
        return jsonify({
            "status": "warning",
            "message": "No file part in the request, but continuing processing"
        }), 200

    file = request.files['images']
    
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({"status": "warning", "message": "No selected file"}), 200
    
    if not file or not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"status": "warning", "message": "Invalid file type"}), 200
    
    try:
        # Read the file content
        file_content = file.read()
        if not file_content:
            logger.warning("Empty file content")
            return jsonify({"status": "warning", "message": "Empty file content"}), 200
            
        img_io = io.BytesIO(file_content)
        
        # Process image and make prediction
        img_array = process_image(img_io)
        predictions = model.predict(img_array)
        
        if predictions.shape[1] != len(Config.CLASS_LABELS):
            logger.error("Model output shape mismatch")
            return jsonify({"error": "Model output shape mismatch"}), 500

        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_class = Config.CLASS_LABELS[class_index]

        if confidence < Config.CONFIDENCE_THRESHOLD:
            predicted_class = "neither"
        
        response = {
            "status": "success",
            "image": secure_filename(file.filename),
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                Config.CLASS_LABELS[i]: float(predictions[0][i]) 
                for i in range(len(Config.CLASS_LABELS))
            }
        }
        
        logger.info(f"Successfully classified image: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    finally:
        logger.info("=== Ending classification request ===")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": Config.MODEL_PATH
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)