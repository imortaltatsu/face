"""
Face detection and preprocessing using MTCNN
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from typing import Optional, Tuple
import config


class FacePreprocessor:
    """Face detection, extraction, and preprocessing"""
    
    def __init__(self):
        """Initialize MTCNN face detector"""
        self.detector = MTCNN()
        print("✓ MTCNN face detector initialized")
    
    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect largest face in image
        
        Args:
            image: RGB image
            
        Returns:
            Detection dict with 'box' and 'keypoints' or None
        """
        detections = self.detector.detect_faces(image)
        
        if not detections:
            return None
        
        # Return largest face
        largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        return largest
    
    def extract_face(self, image: np.ndarray, detection: dict = None, 
                     margin: int = 20) -> Optional[np.ndarray]:
        """
        Extract and crop face from image
        
        Args:
            image: RGB image
            detection: Face detection (auto-detect if None)
            margin: Pixels around face box
            
        Returns:
            Cropped face (160x160x3) or None
        """
        if detection is None:
            detection = self.detect_face(image)
            if detection is None:
                return None
        
        x, y, w, h = detection['box']
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Crop and resize
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, config.INPUT_SHAPE[:2])
        
        return face
    
    def preprocess_for_model(self, face_image: np.ndarray) -> np.ndarray:
        """
        Normalize face for FaceNet
        
        Args:
            face_image: Face image (160x160x3)
            
        Returns:
            Normalized face in range [-1, 1]
        """
        face = face_image.astype(np.float32)
        face = (face - 127.5) / 128.0  # FaceNet normalization
        return face
    
    def process_image(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Complete pipeline: detect -> extract -> preprocess
        
        Args:
            image: RGB image
            
        Returns:
            (preprocessed_face, detection) or None
        """
        detection = self.detect_face(image)
        if detection is None:
            return None
        
        face = self.extract_face(image, detection)
        if face is None:
            return None
        
        face = self.preprocess_for_model(face)
        
        return face, detection
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file (RGB format)"""
        image = Image.open(image_path)
        image = image.convert('RGB')
        return np.array(image)
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Load image from bytes (RGB format)"""
        image = Image.open(image_bytes)
        image = image.convert('RGB')
        return np.array(image)


# Global preprocessor instance
_preprocessor_instance = None

def get_preprocessor() -> FacePreprocessor:
    """Get or create global preprocessor instance"""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = FacePreprocessor()
    return _preprocessor_instance
