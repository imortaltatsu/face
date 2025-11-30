"""
FaceNet model wrapper for face embedding extraction
"""

import numpy as np
from keras_facenet import FaceNet
import config


class FaceNetModel:
    """Wrapper for FaceNet model with embedding extraction"""
    
    def __init__(self):
        """Initialize FaceNet model"""
        print("Loading FaceNet model...")
        self.model = FaceNet()
        self.embedding_size = 512
        print(f"✓ FaceNet loaded (embedding dim: {self.embedding_size})")
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from preprocessed face image
        
        Args:
            face_image: Preprocessed face image (160x160x3), normalized
            
        Returns:
            512-dimensional embedding vector (L2 normalized)
        """
        # Ensure correct shape
        if face_image.shape != (160, 160, 3):
            raise ValueError(f"Expected shape (160, 160, 3), got {face_image.shape}")
        
        # Add batch dimension if needed
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Get embedding
        embedding = self.model.embeddings(face_image)
        
        # L2 normalize
        embedding = self._normalize(embedding[0])
        
        return embedding
    
    def get_embeddings_batch(self, face_images: np.ndarray) -> np.ndarray:
        """
        Extract embeddings for multiple faces
        
        Args:
            face_images: Batch of preprocessed face images (N, 160, 160, 3)
            
        Returns:
            Array of embeddings (N, 512)
        """
        embeddings = self.model.embeddings(face_images)
        
        # L2 normalize each embedding
        normalized = np.array([self._normalize(emb) for emb in embeddings])
        
        return normalized
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding vector
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            L2 normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm


# Global model instance (singleton pattern)
_model_instance = None

def get_model() -> FaceNetModel:
    """Get or create global FaceNet model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = FaceNetModel()
    return _model_instance
