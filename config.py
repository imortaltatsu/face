"""
Configuration settings for face verification system
"""

# Model settings
MODEL_NAME = 'facenet'
EMBEDDING_DIM = 512
INPUT_SHAPE = (160, 160, 3)

# Face detection settings
MIN_FACE_SIZE = 20
DETECTION_CONFIDENCE = 0.9

# Similarity settings
SIMILARITY_METRIC = 'cosine'
VERIFICATION_THRESHOLD = 0.6  # Cosine similarity threshold
IDENTIFICATION_THRESHOLD = 0.5

# Embedding augmentation settings
AUGMENTATION_NOISE_SCALE = 0.01
AUGMENTATION_SAMPLES_PER_IMAGE = 5
EMBEDDING_FUSION_METHOD = 'average'

# User profile settings
PROFILE_STORAGE_PATH = 'data/profiles'
EMBEDDINGS_PER_USER = 5
USE_ENRICHED_EMBEDDINGS = True

# Liveness detection settings
LIVENESS_ENABLED = True
LIVENESS_THRESHOLD = 0.6  # For heuristic-based detection
LIVENESS_USE_ML_MODEL = False  # Disabled - model trained on photos, not webcams
LIVENESS_MOTION_REQUIRED = False  # Disable for single image
LIVENESS_BLINK_REQUIRED = False   # Disable for single image

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Paths
MODEL_CACHE_DIR = 'models/cache'
DATA_DIR = 'data'
UPLOAD_DIR = 'data/uploads'
