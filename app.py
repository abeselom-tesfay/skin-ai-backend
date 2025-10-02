from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model with error handling
try:
    logger.info("Loading TensorFlow model...")
    model = tf.keras.models.load_model('./models/skin_lesion_model_final.keras')
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@app.route('/')
def home():
    return jsonify({'message': 'Skin AI Backend is running!'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if model else 'error',
        'message': 'Model loaded successfully' if model else 'Model failed to load'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # ensure RGB
        
        # Enhanced human skin validation
        skin_check = validate_human_skin(image)
        if not skin_check['is_human_skin']:
            return jsonify({
                'error': 'Please upload human skin images only',
                'detected_content': skin_check['detected_content'],
                'suggestion': skin_check['suggestion'],
                'confidence': skin_check['skin_confidence']
            }), 400
        
        image = image.resize((224, 224))  # model input size
        image_array = np.array(image)
        
        # Fix: ensure shape is (224, 224, 3)
        if image_array.ndim == 2:  # grayscale
            image_array = np.stack([image_array]*3, axis=-1)
        elif image_array.shape[2] == 1:
            image_array = np.concatenate([image_array]*3, axis=-1)
        
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        predictions = model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'prediction': class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': predictions[0].tolist()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def validate_human_skin(image):
    """
    Enhanced validation to detect human skin vs other content
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Multiple skin detection methods
    skin_confidence = 0
    
    # Method 1: Skin color detection in RGB
    lower_skin_rgb = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_rgb = np.array([255, 200, 255], dtype=np.uint8)
    skin_mask_rgb = cv2.inRange(img_array, lower_skin_rgb, upper_skin_rgb)
    skin_ratio_rgb = np.sum(skin_mask_rgb > 0) / (height * width)
    
    # Method 2: Skin color detection in HSV (better for skin)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(img_hsv, lower_skin_hsv, upper_skin_hsv)
    skin_ratio_hsv = np.sum(skin_mask_hsv > 0) / (height * width)
    
    # Method 3: Texture analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Method 4: Edge density (skin has fewer edges than objects/animals)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # Combine confidence scores
    skin_confidence = (skin_ratio_rgb + skin_ratio_hsv) / 2
    
    # Decision logic
    is_human_skin = False
    detected_content = "unknown"
    suggestion = "Please upload a clear image of human skin"
    
    # High confidence skin detection
    if skin_confidence > 0.25 and edge_density < 0.1:
        is_human_skin = True
        detected_content = "Human skin detected"
        suggestion = "Proceeding with analysis..."
    
    # Animal detection (fur, feathers, etc.)
    elif edge_density > 0.15 and skin_confidence < 0.1:
        detected_content = "Animal detected (fur/feathers pattern)"
        suggestion = "This app is for human skin analysis only"
    
    # Object detection (text, objects, etc.)
    elif laplacian_var > 1000 or edge_density > 0.2:
        detected_content = "Object or text detected"
        suggestion = "Please upload human skin images only"
    
    # Low skin confidence
    elif skin_confidence < 0.1:
        detected_content = "No significant skin tones detected"
        suggestion = "Ensure the image contains visible human skin"
    
    # Possible skin but needs verification
    elif skin_confidence < 0.2:
        detected_content = "Limited skin area detected"
        suggestion = "Please upload a clearer image with more visible skin"
    
    # Suspicious patterns (might be fabric, etc.)
    else:
        detected_content = "Unclear content - may not be human skin"
        suggestion = "Upload a clear image of skin lesion on human skin"
    
    return {
        'is_human_skin': is_human_skin,
        'skin_confidence': round(skin_confidence, 3),
        'detected_content': detected_content,
        'suggestion': suggestion,
        'edge_density': round(edge_density, 3),
        'texture_variance': round(laplacian_var, 1)
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
