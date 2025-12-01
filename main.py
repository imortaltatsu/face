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
from anti_spoofing import get_detector
from user_profile import get_database
import config
import shutil
import uuid


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
    description="Face verification system with video-based anti-spoofing and user profiles",
    version="1.1.0",
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
anti_spoofing = None
database = None


def initialize_models():
    """Initialize all models at startup"""
    global model, preprocessor, anti_spoofing, database
    
    print("\n🔄 Initializing models...")
    print("  ⏳ Loading FaceNet model...")
    model = get_model()
    
    print("  ⏳ Loading MTCNN face detector...")
    preprocessor = get_preprocessor()
    
    print("  ⏳ Loading anti-spoofing detector...")
    anti_spoofing = get_detector()
    
    print("  ⏳ Loading DuckDB database...")
    database = get_database()
    
    print("✅ All models initialized and ready!\n")
    return model, preprocessor, anti_spoofing, database


def get_instances():
    """Get initialized model instances"""
    return model, preprocessor, anti_spoofing, database


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
        "version": "1.1.0",
        "features": [
            "Face verification (1:1)",
            "Face identification (1:N)",
            "Video Anti-Spoofing (PAD)",
            "User profile management",
            "Enriched embeddings"
        ]
    }


@app.post("/register", response_model=RegisterResponse)
async def register_user(
    user_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Register a new user with their face image (Legacy image-based)
    """
    try:
        m, prep, _, db = get_instances()
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Extract embedding
        embedding = m.get_embedding(face)
        
        # Create profile
        profile = db.create_profile(user_id, name, embedding)
        
        return RegisterResponse(
            success=True,
            user_id=user_id,
            name=name,
            message=f"User {name} registered successfully"
        )
    
    except Exception as e:
        print(f"❌ Exception in register: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/register_video", response_model=RegisterResponse)
async def register_user_video(
    user_id: str = Form(...),
    name: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Register a new user with a video (includes Anti-Spoofing check)
    """
    try:
        m, prep, detector, db = get_instances()
        
        # Save video temporarily
        temp_filename = f"temp_{uuid.uuid4()}.mp4"
        temp_path = os.path.join(config.UPLOAD_DIR, temp_filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        try:
            # 1. Check Anti-Spoofing
            print(f"🔍 Running anti-spoofing check on {temp_filename}...")
            is_real, score, reason = detector.analyze_video(temp_path)
            
            print(f"📊 Anti-spoofing result: Real={is_real}, Score={score:.4f}, Reason={reason}")
            
            if not is_real:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Anti-spoofing check failed: {reason} (Score: {score:.2f})"
                )
            
            # 2. Extract face from video for registration
            # We need to get a good frame. The detector already extracted frames.
            # For simplicity, we'll extract the first valid frame again or use the video processing.
            # Let's use OpenCV to get the middle frame
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise HTTPException(status_code=400, detail="Could not extract frame from video")
                
            # Convert BGR to RGB
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face
            result = prep.process_image(img_array)
            if result is None:
                raise HTTPException(status_code=400, detail="No face detected in video")
            
            face, detection = result
            
            # Extract embedding
            embedding = m.get_embedding(face)
            
            # Create profile
            profile = db.create_profile(user_id, name, embedding)
            
            return RegisterResponse(
                success=True,
                user_id=user_id,
                name=name,
                message=f"User {name} registered with video (Anti-spoofing score: {score:.2f})"
            )
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exception in register_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video registration failed: {str(e)}")


@app.post("/verify", response_model=VerifyResponse)
async def verify_user(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Verify user with image (Legacy, no anti-spoofing)
    """
    try:
        m, prep, _, db = get_instances()
        
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
        
        # Extract embedding
        embedding = m.get_embedding(face)
        
        # Verify against user's embedding
        user_embedding = profile.get_embedding()
        is_match, similarity = verify_faces(embedding, user_embedding)
        
        return VerifyResponse(
            success=True,
            is_match=is_match,
            similarity=similarity,
            confidence=similarity,
            liveness_passed=True, # Skipped for image
            liveness_score=0.0,
            liveness_reason="Image verification (liveness skipped)",
            message=f"Verification {'successful' if is_match else 'failed'}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/verify_video", response_model=VerifyResponse)
async def verify_user_video(
    user_id: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Verify user with video (includes Anti-Spoofing)
    """
    try:
        m, prep, detector, db = get_instances()
        
        # Get user profile
        profile = db.get_profile(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        # Save video temporarily
        temp_filename = f"temp_verify_{uuid.uuid4()}.mp4"
        temp_path = os.path.join(config.UPLOAD_DIR, temp_filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        try:
            # 1. Check Anti-Spoofing
            print(f"🔍 Running anti-spoofing check on {temp_filename}...")
            is_real, score, reason = detector.analyze_video(temp_path)
            
            print(f"📊 Anti-spoofing result: Real={is_real}, Score={score:.4f}, Reason={reason}")
            
            if not is_real:
                return VerifyResponse(
                    success=False,
                    is_match=False,
                    similarity=0.0,
                    confidence=0.0,
                    liveness_passed=False,
                    liveness_score=score,
                    liveness_reason=reason,
                    message=f"Anti-spoofing failed: {reason}"
                )
            
            # 2. Extract face for verification
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise HTTPException(status_code=400, detail="Could not extract frame from video")
                
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face
            result = prep.process_image(img_array)
            if result is None:
                raise HTTPException(status_code=400, detail="No face detected in video")
            
            face, detection = result
            
            # Extract embedding
            embedding = m.get_embedding(face)
            
            # Verify
            user_embedding = profile.get_embedding()
            is_match, similarity = verify_faces(embedding, user_embedding)
            
            return VerifyResponse(
                success=True,
                is_match=is_match,
                similarity=similarity,
                confidence=(similarity + score) / 2,
                liveness_passed=True,
                liveness_score=score,
                liveness_reason=reason,
                message=f"Verification {'successful' if is_match else 'failed'}"
            )
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exception in verify_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video verification failed: {str(e)}")


@app.post("/identify", response_model=IdentifyResponse)
async def identify_user(image: UploadFile = File(...)):
    """
    Identify which registered user matches the face in the image (1:N matching)
    """
    try:
        m, prep, _, db = get_instances()
        
        # Read and process image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert('RGB'))
        
        # Process face
        result = prep.process_image(img_array)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face, detection = result
        
        # Extract embedding
        embedding = m.get_embedding(face)
        
        # Identify from database
        all_embeddings = db.get_all_embeddings()
        
        if not all_embeddings:
            return IdentifyResponse(
                success=True,
                identified=False,
                user_id=None,
                name=None,
                similarity=0.0,
                liveness_passed=True,
                liveness_score=0.0,
                message="No users registered in database"
            )
        
        user_id, similarity = identify_face(embedding, all_embeddings)
        
        if user_id:
            profile = db.get_profile(user_id)
            return IdentifyResponse(
                success=True,
                identified=True,
                user_id=user_id,
                name=profile.name,
                similarity=similarity,
                liveness_passed=True,
                liveness_score=0.0,
                message=f"Identified as {profile.name}"
            )
        else:
            return IdentifyResponse(
                success=True,
                identified=False,
                user_id=None,
                name=None,
                similarity=0.0,
                liveness_passed=True,
                liveness_score=0.0,
                message="No match found"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@app.post("/add-face/{user_id}")
async def add_face_to_profile(user_id: str, image: UploadFile = File(...)):
    """
    Add additional face image to existing user profile
    """
    try:
        m, prep, _, db = get_instances()
        
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
        
        # Extract embedding
        embedding = m.get_embedding(face)
        
        # Update profile
        db.update_profile(user_id, embedding)
        
        return {
            "success": True,
            "message": f"Face added to {user_id}'s profile.",
            "liveness_score": 0.0
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
    Compare two face images directly
    """
    try:
        m, prep, _, _ = get_instances()
        
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
        
        # Extract embeddings
        emb1 = m.get_embedding(face1)
        emb2 = m.get_embedding(face2)
        
        # Compare
        is_match, similarity = verify_faces(emb1, emb2)
        
        return {
            "success": True,
            "is_match": is_match,
            "similarity": similarity,
            "both_live": True # Skipped
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
    print("  ✓ Video Anti-Spoofing (PAD)")
    print("  ✓ User profile management with DuckDB")
    print("  ✓ Enriched embeddings via vector addition")
    print("  ✓ FaceNet embeddings (512-dim)")
    print("  ✓ Models preloaded at startup")
    print(f"\nAPI Documentation: http://localhost:{config.API_PORT}/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
