import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from tensorflow.keras.models import load_model
import logging
import sys

# Suppress TensorFlow warnings for faster startup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

logger.info("Starting Emotion Detection Backend...")

# Initialize face cascade classifier
logger.info("Loading Haar Cascade...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logger.info("Haar Cascade loaded")

class_names = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad']
img_height, img_width = 48, 48
MODEL_DIR = os.path.abspath("models/face")
MODEL_FILENAME = 'Face_Emotion1.h5'

# Load the trained model at startup
logger.info("Loading emotion detection model (this may take 30-60 seconds)...")
try:
    model = load_model(os.path.join(MODEL_DIR, MODEL_FILENAME), compile=False)
    logger.info("Model loaded successfully!")
    
    # Warm up the model with a dummy prediction
    logger.info("Warming up model...")
    dummy_input = np.zeros((1, 48, 48, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    logger.info("Model warmed up and ready!")
    logger.info("✅ Application ready to accept requests")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    model = None
def preprocess_image(img_array):
    """
    Preprocess the image for emotion detection:
    1. Detect Face (Haar Cascade)
    2. Convert to RGB
    3. Resize to 48x48 (matching model input)
    4. Normalize (/255.0)
    """
    try:
        # Convert to Gray for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If face detected, crop it. If not, resize the whole image.
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img_array[y:y + h, x:x + w]
        else:
            face_roi = img_array  # Fallback to full image

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Resize to model input shape (48, 48)
        face_resized = cv2.resize(face_rgb, (img_width, img_height))

        # Normalize
        face_normalized = face_resized.astype('float32') / 255.0

        # Expand dims: (48,48,3) -> (1,48,48,3)
        input_data = np.expand_dims(face_normalized, axis=0)

        return input_data
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    try:
        logger.info("Received image for emotion detection")
        
        # Read the file content once and store it in memory
        file_content = image_file.read()

        if not file_content:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        # Convert to numpy array for OpenCV
        image_bytes = np.frombuffer(file_content, np.uint8)

        # Decode the image
        img_array = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img_array is None:
            return jsonify({"error": "Invalid or unsupported image format."}), 400

        # Preprocess the image
        input_data = preprocess_image(img_array)

        if input_data is None:
            return jsonify({"error": "Error processing image"}), 400

        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(input_data, verbose=0)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        emotion = class_names[class_idx]
        
        logger.info(f"Prediction: {emotion} ({confidence:.2f}%)")

        return jsonify({
            "Emotion": emotion,
            "confidence": float(confidence)
        })

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/detect_with_questionnaire', methods=['POST'])
def detect_with_questionnaire():
    """
    Enhanced emotion detection with questionnaire scoring
    Expected JSON payload:
    {
        "questionnaire_score": 8,  // Score from 5 questions (0-10)
        "image": <file>
    }
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Get questionnaire score from form data
    try:
        questionnaire_score = float(request.form.get('questionnaire_score', 0))
        if not (0 <= questionnaire_score <= 10):
            return jsonify({"error": "Questionnaire score must be between 0 and 10"}), 400
    except ValueError:
        return jsonify({"error": "Invalid questionnaire score format"}), 400

    image_file = request.files['image']

    try:
        # Read and process image
        file_content = image_file.read()
        if not file_content:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        image_bytes = np.frombuffer(file_content, np.uint8)
        img_array = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img_array is None:
            return jsonify({"error": "Invalid or unsupported image format."}), 400

        # Preprocess and predict emotion
        input_data = preprocess_image(img_array)
        if input_data is None:
            return jsonify({"error": "Error processing image"}), 400

        predictions = model.predict(input_data)
        class_idx = np.argmax(predictions)
        emotion_confidence = np.max(predictions) * 100
        emotion = class_names[class_idx]

        # Calculate combined score: (Questionnaire + Emotion Confidence/10) / 2 * 10
        # Normalize emotion confidence to 0-10 scale
        emotion_score = emotion_confidence / 10
        combined_score = (questionnaire_score + emotion_score) / 2

        # Determine risk level based on combined score
        if combined_score >= 8:
            risk_level = "Normal"
        elif combined_score >= 4:
            risk_level = "Low Risk"
        else:
            risk_level = "High Risk"

        return jsonify({
            "emotion": emotion,
            "emotion_confidence": float(emotion_confidence),
            "questionnaire_score": float(questionnaire_score),
            "combined_score": float(round(combined_score, 2)),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Verify model is loaded
        if model is not None:
            return jsonify({
                "status": "healthy",
                "service": "emotion-detection-backend",
                "version": "1.0.0",
                "model_loaded": True
            }), 200
        else:
            return jsonify({
                "status": "unhealthy",
                "service": "emotion-detection-backend",
                "error": "Model not loaded"
            }), 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
