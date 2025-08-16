from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import uuid
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from werkzeug.utils import secure_filename
from config import Config
from services.prediction_service import PredictionService
from utils.validators import validate_images
from utils.helpers import cleanup_old_files
import logging

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Add this new route for the change detection page
@app.route('/change-detection')
def change_detection():
    return render_template('ntend.html')

# Initialize prediction service
prediction_service = PredictionService()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Change detection API is running'})

@app.route('/detect-changes', methods=['POST'])
def detect_changes():
    from PIL import Image
    import numpy as np
    """Main endpoint for change detection"""
    try: 
        # Validate request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both image1 and image2 are required'}), 400
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        # Validate images
        validation_result = validate_images([image1, image2])
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded images temporarily
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_img1.jpg")
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_img2.jpg")
        
        image1.save(image1_path)
        image2.save(image2_path)
        
        # Process images and get prediction
        result_path = prediction_service.predict_changes(image1_path, image2_path, session_id)
        # Calculate percentage change
        try:
             mask_img = Image.open(result_path).convert("L")  # ensure grayscale
             mask_array = np.array(mask_img)
             changed_pixels = np.count_nonzero(mask_array)  # non-black pixels
             total_pixels = mask_array.size
             percentage_change = (changed_pixels / total_pixels) * 100
        except Exception as calc_err:
             logger.error(f"Error calculating percentage change: {calc_err}")
             percentage_change = None

        
        # Clean up input images
        os.remove(image1_path)
        os.remove(image2_path)
        
        return jsonify({
    'success': True,
    'session_id': session_id,
    'result_url': f'/get-result/{session_id}',
    'percentage_change': round(percentage_change, 2) if percentage_change is not None else None
})

        
    except Exception as e:
        logger.error(f"Error in change detection: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/get-result/<session_id>', methods=['GET'])
def get_result(session_id):
    """Get the binary result image"""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_result.png")
        
        if not os.path.exists(result_path):
            return jsonify({'error': 'Result not found'}), 404
        
        return send_file(result_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error serving result: {str(e)}")
        return jsonify({'error': 'Error serving result'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old temporary files"""
    try:
        cleanup_old_files(app.config['UPLOAD_FOLDER'], hours=1)
        cleanup_old_files(app.config['RESULTS_FOLDER'], hours=24)
        return jsonify({'success': True, 'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)