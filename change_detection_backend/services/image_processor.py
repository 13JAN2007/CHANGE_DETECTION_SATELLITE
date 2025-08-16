import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from config import Config

class ImageProcessor:
    def __init__(self):
        # Transform to match your model's training preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),  # 512x512
            transforms.ToTensor(),  # Converts to [0,1] range
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input - matches your training code"""
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
    def postprocess_output(self, model_output):
        """Convert model output to binary PIL Image - matches your prediction code"""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(model_output)
        
        # Apply threshold to make it binary (same as your predict.py)
        binary_tensor = (probs > 0.5).float()
        
        # Convert to numpy and scale to 0-255
        binary_np = binary_tensor.squeeze().cpu().numpy() * 255
        
        # Convert to PIL Image
        binary_image = Image.fromarray(binary_np.astype(np.uint8), mode='L')
        
        return binary_image