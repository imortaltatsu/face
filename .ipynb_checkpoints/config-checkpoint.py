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

# Anti-spoofing settings (Video-based)
ANTI_SPOOFING_ENABLED = True
ANTI_SPOOFING_MODEL = 'models/anti_spoofing_int8.tflite'
SEQUENCE_LENGTH = 30
VIDEO_FPS = 10
MAX_VIDEO_SIZE_MB = 50
ANTI_SPOOFING_THRESHOLD = 0.5

# Legacy Liveness detection settings (Image-based)
LIVENESS_ENABLED = False
LIVENESS_THRESHOLD = 0.4
LIVENESS_USE_ML_MODEL = False
LIVENESS_MOTION_REQUIRED = False
LIVENESS_BLINK_REQUIRED = False

# API settings
API_HOST = "0.0.0.0"
API_PORT = 9834  # Production server port
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Paths
MODEL_CACHE_DIR = 'models/cache'
DATA_DIR = 'data'
UPLOAD_DIR = 'data/uploads'
