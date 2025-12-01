"""
FastAPI backend for face verification system with liveness detection
"""

import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from PIL import Image

# Import our modules
from model import get_model
from preprocessing import get_preprocessor
from similarity import verify_faces, identify_face, cosine_similarity
from liveness import get_liveness_detector
from user_profile import get_database
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup: Initialize models
    initialize_models()
    yield
    # Shutdown: cleanup if needed
    pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Face Verification API",
    description="Face verification system with liveness detection and user profiles",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (preloaded at startup)
model = None
preprocessor = None
liveness_detector = None
database = None


def initialize_models():
    """Initialize all models at startup"""
    global model, preprocessor, liveness_detector, database
    
    print("\n🔄 Initializing models...")
    print("  ⏳ Loading FaceNet model...")
    model = get_model()
    
    print("  ⏳ Loading MTCNN face detector...")
    preprocessor = get_preprocessor()
    
    print("  ⏳ Loading liveness detector...")
    liveness_detector = get_liveness_detector()
    
    print("  ⏳ Loading DuckDB database...")
    database = get_database()
    
    print("✅ All models initialized and ready!\n")
    return model, preprocessor, liveness_detector, database


def get_instances():
    """Get initialized model instances"""
    return model, preprocessor, liveness_detector, database


# Pydantic models for request/response
class VerifyRequest(BaseModel):
    user_id: str


class VerifyResponse(BaseModel):
    success: bool
    is_match: bool
    similarity: float
    confidence: float
    liveness_passed: bool
    liveness_score: float
    liveness_reason: str
    message: str


class RegisterResponse(BaseModel):
    success: bool
    user_id: str
    name: str
    message: str


class IdentifyResponse(BaseModel):
    success: bool
    identified: bool
    user_id: Optional[str]
    name: Optional[str]
    similarity: float
    liveness_passed: bool
    liveness_score: float
    message: str


class UserListResponse(BaseModel):
    success: bool
    users: List[dict]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Face Verification API",
        "version": "1.0.0",
        "features": [
            "Face verification (1:1)",
            "Face identification (1:N)",
            "Liveness detection",
            "User profile management",
            "Enriched embeddings with vector addition"
        ]
    }


@app.post("/register", response_model=RegisterResponse)
async def register_user(
    user_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Register a new user with their face image
    
    - **user_id**: Unique user identifier
    - **name**: User's name
    - **image**: Face image file
    """
    try:
        m, prep, liveness, db = get_instances()
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Check liveness
        print(f"🔍 Calling liveness detection...")
        print(f"   Image shape: {img_array.shape}")
        print(f"   Detection box: {detection}")
        
        is_live, liveness_score, reason = liveness.detect_liveness(img_array, detection)
        
        print(f"📊 Liveness result:")
        print(f"   is_live: {is_live}")
        print(f"   score: {liveness_score}")
        print(f"   reason: {reason}")
        
        if not is_live:
            raise HTTPException(
                status_code=400, 
                detail=f"Liveness check failed: {reason} (score: {liveness_score:.2f})"
            )
        
        # Extract embedding
        embedding = m.extract_embedding(face)
        
        # Create profile
        profile = db.create_profile(user_id, name, embedding)
        
        return RegisterResponse(
            success=True,
            user_id=user_id,
            name=name,
            message=f"User {name} registered successfully with liveness score {liveness_score:.2f}"
        )
    
    except ValueError as e:
        print(f"❌ ValueError in register: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Exception in register: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/verify", response_model=VerifyResponse)
async def verify_user(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Verify if the face in the image matches the registered user
    
    - **user_id**: User ID to verify against
    - **image**: Face image file
    """
    try:
        m, prep, liveness, db = get_instances()
        
        # Get user profile
        profile = db.get_profile(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Check liveness
        is_live, liveness_score, reason = liveness.detect_liveness(img_array, detection)
        
        # Extract embedding
        embedding = m.extract_embedding(face)
        
        # Verify against user's embedding
        user_embedding = profile.get_embedding()
        is_match, similarity = verify_faces(embedding, user_embedding)
        
        # Calculate overall confidence
        confidence = (similarity + liveness_score) / 2.0 if is_live else similarity * 0.5
        
        return VerifyResponse(
            success=True,
            is_match=is_match and is_live,
            similarity=similarity,
            confidence=confidence,
            liveness_passed=is_live,
            liveness_score=liveness_score,
            liveness_reason=reason,
            message=f"Verification {'successful' if (is_match and is_live) else 'failed'}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/identify", response_model=IdentifyResponse)
async def identify_user(image: UploadFile = File(...)):
    """
    Identify which registered user matches the face in the image (1:N matching)
    
    - **image**: Face image file
    """
    try:
        m, prep, liveness, db = get_instances()
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Check liveness
        is_live, liveness_score, reason = liveness.detect_liveness(img_array, detection)
        
        # Extract embedding
        embedding = m.extract_embedding(face)
        
        # Identify from database
        all_embeddings = db.get_all_embeddings()
        
        if not all_embeddings:
            return IdentifyResponse(
                success=True,
                identified=False,
                user_id=None,
                name=None,
                similarity=0.0,
                liveness_passed=is_live,
                liveness_score=liveness_score,
                message="No users registered in database"
            )
        
        user_id, similarity = identify_face(embedding, all_embeddings)
        
        if user_id and is_live:
            profile = db.get_profile(user_id)
            return IdentifyResponse(
                success=True,
                identified=True,
                user_id=user_id,
                name=profile.name,
                similarity=similarity,
                liveness_passed=is_live,
                liveness_score=liveness_score,
                message=f"Identified as {profile.name}"
            )
        else:
            return IdentifyResponse(
                success=True,
                identified=False,
                user_id=None,
                name=None,
                similarity=similarity if user_id else 0.0,
                liveness_passed=is_live,
                liveness_score=liveness_score,
                message=f"No match found" if user_id is None else f"Liveness check failed: {reason}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@app.post("/add-face/{user_id}")
async def add_face_to_profile(user_id: str, image: UploadFile = File(...)):
    """
    Add additional face image to existing user profile (for enriched embeddings)
    
    - **user_id**: User ID
    - **image**: Additional face image
    """
    try:
        m, prep, liveness, db = get_instances()
        
        # Get user profile
        profile = db.get_profile(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Check liveness
        is_live, liveness_score, reason = liveness.detect_liveness(img_array, detection)
        if not is_live:
            raise HTTPException(status_code=400, detail=f"Liveness check failed: {reason}")
        
        # Extract embedding
        embedding = m.extract_embedding(face)
        
        # Update profile (will create enriched embedding automatically)
        db.update_profile(user_id, embedding)
        
        return {
            "success": True,
            "message": f"Face added to {user_id}'s profile. Enriched embedding updated.",
            "liveness_score": liveness_score
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add face: {str(e)}")


@app.get("/users", response_model=UserListResponse)
async def list_users():
    """List all registered users"""
    try:
        _, _, _, db = get_instances()
        users = db.list_profiles()
        
        return UserListResponse(
            success=True,
            users=users
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get user profile details"""
    try:
        _, _, _, db = get_instances()
        profile = db.get_profile(user_id)
        
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {
            "success": True,
            "user_id": profile.user_id,
            "name": profile.name,
            "num_embeddings": len(profile.embeddings),
            "has_enriched_embedding": profile.enriched_embedding is not None,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete a user profile"""
    try:
        _, _, _, db = get_instances()
        profile = db.get_profile(user_id)
        
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        db.delete_profile(user_id)
        
        return {
            "success": True,
            "message": f"User {user_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


@app.post("/compare")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Compare two face images directly (no user profile needed)
    
    - **image1**: First face image
    - **image2**: Second face image
    """
    try:
        m, prep, liveness, _ = get_instances()
        
        # Process first image
        img1_bytes = await image1.read()
        img1 = np.array(Image.open(io.BytesIO(img1_bytes)).convert('RGB'))
        result1 = prep.process_image(img1)
        if result1 is None:
            raise HTTPException(status_code=400, detail="No face detected in first image")
        face1, det1 = result1
        
        # Process second image
        img2_bytes = await image2.read()
        img2 = np.array(Image.open(io.BytesIO(img2_bytes)).convert('RGB'))
        result2 = prep.process_image(img2)
        if result2 is None:
            raise HTTPException(status_code=400, detail="No face detected in second image")
        face2, det2 = result2
        
        # Check liveness for both
        live1, score1, reason1 = liveness.detect_liveness(img1, det1)
        live2, score2, reason2 = liveness.detect_liveness(img2, det2)
        
        # Extract embeddings
        emb1 = m.extract_embedding(face1)
        emb2 = m.extract_embedding(face2)
        
        # Compare
        is_match, similarity = verify_faces(emb1, emb2)
        
        return {
            "success": True,
            "is_match": is_match,
            "similarity": similarity,
            "image1_liveness": {"passed": live1, "score": score1, "reason": reason1},
            "image2_liveness": {"passed": live2, "score": score2, "reason": reason2},
            "both_live": live1 and live2
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


def main():
    """Run the FastAPI server"""
    import uvicorn
    
    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.PROFILE_STORAGE_PATH, exist_ok=True)
    
    print("=" * 60)
    print("Face Verification API Server")
    print("=" * 60)
    print(f"Starting server at http://{config.API_HOST}:{config.API_PORT}")
    print("\nFeatures:")
    print("  ✓ Face verification (1:1 matching)")
    print("  ✓ Face identification (1:N matching)")
    print("  ✓ Liveness detection (anti-spoofing)")
    print("  ✓ User profile management with DuckDB")
    print("  ✓ Enriched embeddings via vector addition")
    print("  ✓ FaceNet embeddings (512-dim)")
    print("  ✓ Models preloaded at startup")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
