from config import Config
import os

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_images(files):
    """Validate uploaded image files"""
    for file in files:
        if file.filename == '':
            return {'valid': False, 'message': 'No file selected'}
        
        if not allowed_file(file.filename):
            return {'valid': False, 'message': f'Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}'}
    
    return {'valid': True, 'message': 'Valid images'}