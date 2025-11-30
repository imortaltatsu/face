# Face Verification System

A production-ready face verification system with **liveness detection**, **enriched embeddings**, and **user profile management** using FaceNet, FastAPI, and DuckDB.

## Features

✅ **Face Verification (1:1)** - Verify if a face matches a registered user  
✅ **Face Identification (1:N)** - Identify which user a face belongs to  
✅ **Liveness Detection** - Distinguish real faces from photos/screens using texture analysis  
✅ **Enriched Embeddings** - Vector addition to create robust face representations  
✅ **User Profiles** - DuckDB storage for efficient user management  
✅ **FaceNet Embeddings** - 512-dimensional embeddings (more accurate than MobileFaceNet)  
✅ **FastAPI Backend** - RESTful API with automatic documentation  

## Architecture

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Embeddings**: FaceNet (Inception-ResNet-V1) - 512 dimensions
- **Similarity Metric**: Cosine similarity with L2-normalized embeddings
- **Liveness Detection**: Texture analysis + color diversity + moiré pattern detection
- **Database**: DuckDB for efficient embedding storage and retrieval
- **Enrichment**: Gaussian noise augmentation + vector fusion

## Installation

```bash
# Clone the repository
cd /Users/aditya/proj/face

# Install dependencies (using uv)
uv sync

# Or manually install
uv add tensorflow keras-facenet mtcnn numpy pillow opencv-python fastapi uvicorn python-multipart duckdb
```

## Quick Start

### 1. Start the API Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 2. Register a User

```bash
curl -X POST "http://localhost:8000/register" \
  -F "user_id=user123" \
  -F "name=John Doe" \
  -F "image=@/path/to/face.jpg"
```

Response:
```json
{
  "success": true,
  "user_id": "user123",
  "name": "John Doe",
  "message": "User John Doe registered successfully with liveness score 0.85"
}
```

### 3. Verify a User

```bash
curl -X POST "http://localhost:8000/verify" \
  -F "user_id=user123" \
  -F "image=@/path/to/verification_image.jpg"
```

Response:
```json
{
  "success": true,
  "is_match": true,
  "similarity": 0.92,
  "confidence": 0.88,
  "liveness_passed": true,
  "liveness_score": 0.85,
  "liveness_reason": "Live face detected",
  "message": "Verification successful"
}
```

### 4. Identify a User (1:N)

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "image=@/path/to/unknown_face.jpg"
```

Response:
```json
{
  "success": true,
  "identified": true,
  "user_id": "user123",
  "name": "John Doe",
  "similarity": 0.89,
  "liveness_passed": true,
  "liveness_score": 0.82,
  "message": "Identified as John Doe"
}
```

### 5. Add Additional Face Images (Enriched Embeddings)

```bash
curl -X POST "http://localhost:8000/add-face/user123" \
  -F "image=@/path/to/another_face.jpg"
```

This creates an **enriched embedding** by combining multiple face images using vector addition.

### 6. Compare Two Faces Directly

```bash
curl -X POST "http://localhost:8000/compare" \
  -F "image1=@/path/to/face1.jpg" \
  -F "image2=@/path/to/face2.jpg"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/register` | POST | Register new user with face image |
| `/verify` | POST | Verify face against registered user |
| `/identify` | POST | Identify user from face (1:N) |
| `/add-face/{user_id}` | POST | Add additional face to user profile |
| `/users` | GET | List all registered users |
| `/users/{user_id}` | GET | Get user profile details |
| `/users/{user_id}` | DELETE | Delete user profile |
| `/compare` | POST | Compare two face images |

## Configuration

Edit `config.py` to customize:

```python
# Model settings
VERIFICATION_THRESHOLD = 0.6  # Cosine similarity threshold
IDENTIFICATION_THRESHOLD = 0.5

# Liveness detection
LIVENESS_THRESHOLD = 0.7
LIVENESS_ENABLED = True

# Embedding enrichment
USE_ENRICHED_EMBEDDINGS = True
EMBEDDINGS_PER_USER = 5
AUGMENTATION_NOISE_SCALE = 0.01

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

## How It Works

### 1. Face Detection & Preprocessing
- MTCNN detects faces and facial landmarks
- Face is aligned and cropped to 160x160
- Normalized to [-1, 1] range for FaceNet

### 2. Embedding Extraction
- FaceNet generates 512-dimensional embedding
- L2 normalization for cosine similarity

### 3. Liveness Detection
- **Texture Analysis**: Real faces have higher Laplacian variance
- **Color Diversity**: Real skin has more color variation than prints
- **Moiré Pattern**: Screens show periodic patterns in FFT

### 4. Enriched Embeddings (Vector Addition)
- Multiple face images per user are stored
- Gaussian noise is added to create synthetic variations
- All embeddings are fused using weighted average
- Result: More robust face representation

### 5. Verification & Identification
- **1:1 Verification**: Compare query embedding with user's enriched embedding
- **1:N Identification**: Find best match from all users in database
- Cosine similarity used for comparison

## Project Structure

```
face/
├── main.py                      # FastAPI backend
├── config.py                    # Configuration settings
├── model.py                     # FaceNet model loader
├── preprocessing.py             # Face detection & preprocessing
├── similarity.py                # Cosine similarity & verification
├── liveness.py                  # Liveness detection
├── embedding_augmentation.py    # Vector addition & enrichment
├── user_profile.py              # DuckDB user management
├── data/
│   ├── profiles/                # DuckDB database
│   └── uploads/                 # Temporary uploads
└── pyproject.toml               # uv dependencies
```

## Performance

- **FaceNet**: 99.63% accuracy on LFW dataset
- **Embedding Size**: 512 dimensions
- **Inference Speed**: ~100ms per face (CPU)
- **Liveness Detection**: ~50ms per image

## Why FaceNet over MobileFaceNet?

- **Higher Accuracy**: FaceNet achieves 99.63% vs MobileFaceNet's ~99.2%
- **Better Embeddings**: 512-dim embeddings capture more facial details
- **Proven Track Record**: Industry standard for face verification
- **TFLite Compatible**: Can be converted for mobile deployment

## Future Enhancements

- [ ] TFLite conversion for mobile deployment
- [ ] Video-based liveness detection (blink detection, head movement)
- [ ] Fine-tuning on custom dataset
- [ ] Triplet loss training for domain adaptation
- [ ] Face clustering and deduplication
- [ ] Multi-face tracking in video streams

## License

MIT

## Credits

- **FaceNet**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- **MTCNN**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
# face
