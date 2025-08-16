import os
import torch
import numpy as np
from PIL import Image
from models.change_detection_model import UNet
from services.image_processor import ImageProcessor
from config import Config
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.image_processor = ImageProcessor()
        self.device = Config.DEVICE
        self._load_model()
    
    def _load_model(self):
        """Load the trained U-Net model"""
        try:
            self.model = UNet(in_channels=6)
            self.model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("U-Net model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_changes(self, image1_path, image2_path, session_id):
        """Predict changes between two images"""
        try:
            # Preprocess images
            img1_tensor = self.image_processor.preprocess_image(image1_path)
            img2_tensor = self.image_processor.preprocess_image(image2_path)
            
            # Concatenate images along channel dimension (6 channels total)
            input_tensor = torch.cat([img1_tensor, img2_tensor], dim=0)
            input_tensor = input_tensor.to(self.device).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                
            # Post-process output to binary image
            binary_image = self.image_processor.postprocess_output(logits)
            
            # Save result
            result_path = os.path.join(Config.RESULTS_FOLDER, f"{session_id}_result.png")
            binary_image.save(result_path)
            
            return result_path
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise