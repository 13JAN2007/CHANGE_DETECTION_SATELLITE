import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads/temp'
    RESULTS_FOLDER = 'uploads/results'
    MODEL_PATH = 'models/saved_models/best_cd_model.pth'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Model specific settings (based on your model)
    IMAGE_SIZE = (512, 512)  # Your model uses 512x512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'