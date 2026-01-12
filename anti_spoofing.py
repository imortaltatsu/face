"""
Robust Face Anti-Spoofing / Liveness Detection Module.
Uses TFLite model on strictly cropped face regions.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
import logging
from typing import Tuple, Optional, List
import config

logger = logging.getLogger(__name__)

class LivenessDetector:
    """
    Motion-based liveness detector.
    Checks for head pose variance (motion) to distinguish 3D live faces from 2D static attacks.
    """
    
    def __init__(self):
        # No model loading needed for purely heuristic liveness
        self.input_shape = (224, 224) 
        logger.info("✓ Motion-based Liveness Detector ready (No DL model)")

    def _analyze_head_pose(self, landmarks_list: List[dict]) -> float:
        """
        Analyze variance in head pose (yaw/pitch) to detect static 2D attacks.
        Uses rotation-invariant vector projection to distinguish true 3D rotation 
        from 2D rotation/shaking (which is a common spoofing technique).
        
        Returns variance score (higher = more 3D movement = likely real).
        """
        if len(landmarks_list) < 5:
            return 0.0
            
        # We will track the relative position of the nose with respect to the eyes
        # This is invariant to 2D translation, rotation, and scale.
        yaw_ratios = []
        pitch_ratios = []
        
        for lm in landmarks_list:
            # Convert to numpy for vector math
            # Landmarks: left_eye, right_eye, nose
            l_eye = np.array(lm['left_eye'])
            r_eye = np.array(lm['right_eye'])
            nose = np.array(lm['nose'])
            
            # 1. Define the coordinate system based on eyes (invariant to head roll/size)
            eye_vec = r_eye - l_eye
            eye_dist = np.linalg.norm(eye_vec)
            
            if eye_dist < 1e-6:
                continue
                
            # Unit vector along the eye line (X-axis of face)
            u_x = eye_vec / eye_dist
            
            # Unit vector perpendicular to eye line (Y-axis of face)
            # Rotate 90 degrees: (x, y) -> (-y, x) assuming image coords (y down)
            u_y = np.array([-u_x[1], u_x[0]])
            
            # 2. Project nose position onto this coordinate system relative to left eye
            nose_vec = nose - l_eye
            
            # Projections
            proj_x = np.dot(nose_vec, u_x)
            proj_y = np.dot(nose_vec, u_y)
            
            # Normalize by scale (eye distance)
            rel_x = proj_x / eye_dist
            rel_y = proj_y / eye_dist
            
            yaw_ratios.append(rel_x)
            pitch_ratios.append(rel_y)
            
        if not yaw_ratios:
            return 0.0
            
        # Calculate variance
        # High variance in these ratios means the nose is moving relative to the eyes 
        # in a way that implies 3D structure changes (turning head).
        # 2D rotation of a photo keeps these ratios CONSTANT.
        var_yaw = np.var(yaw_ratios)
        var_pitch = np.var(pitch_ratios)
        
        # Combined score (weighted)
        # Yaw is usually more dominant in liveness checks, but pitch helps too.
        total_variance = var_yaw + var_pitch
        
        return total_variance

    def _analyze_image_quality(self, face_img: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze texture (sharpness), glare, and color distribution.
        Returns: (texture_score, glare_score, color_score)
        """
        if face_img is None or face_img.size == 0:
            return 0.0, 0.0, 0.0
            
        # 1. Texture / Sharpness (Laplacian Variance)
        # Real faces have organic texture. 
        # Low score = Blurry (Print/Low-res Screen).
        # Extremely high score + noise = Grainy Screen.
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Glare Detection (Specular Highlights)
        # Screens have glass surfaces that reflect light sharply.
        # We look for pixels that are near pure white in luminance.
        # Threshold 230/255 is safer for lighter skin tones vs actual glare.
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = np.count_nonzero(bright_mask) / (gray.shape[0] * gray.shape[1])
        
        # 3. Color Analysis (HSV)
        # Screens often have fewer unique colors (quantization) or unnatural saturation.
        hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Saturation histogram analysis
        # Real skin usually has a well-distributed saturation.
        # Screens might be oversaturated or washed out (low variance).
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # Value (Brightness) analysis
        # Screens emit light, often leading to clipped highlights or lifted shadows.
        v_hist = cv2.calcHist([v], [0], None, [256], [0, 256])
        # Check for clipping at the top end (255)
        clipping_ratio = v_hist[255][0] / v.size
        
        # Combine into a "suspicion" score for color
        # High clipping or very low saturation variance is suspicious
        color_suspicion = clipping_ratio + (1.0 if s_std < 10 else 0.0)
        
        return texture_score, glare_ratio, color_suspicion

    def analyze_video(self, video_path: str, preprocessor) -> Tuple[bool, float, str, Optional[np.ndarray]]:
        """
        Process video: Extract faces -> Check Motion Liveness -> Aggregate
        Returns: (is_real, score, reason, best_frame)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, 0.0, "Could not open video file", None
            
        # Analyze strategy
        stride = 2
        
        landmarks_list = []
        best_frame = None
        max_face_area = 0
        frames_processed = 0
        
        # Quality metrics accumulators
        total_texture = 0
        total_glare = 0
        total_color_suspicion = 0
        valid_frames = 0
        
        logger.info(f"Analyzing video {video_path} for liveness...")
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if current_frame % stride != 0:
                current_frame += 1
                continue

            frames_processed += 1
            current_frame += 1
            
            # Limit max frames
            if len(landmarks_list) >= 40:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            detection = preprocessor.detect_face(frame_rgb)
            if detection is None:
                continue
            
            # Extract face for quality analysis
            face_crop = preprocessor.extract_face(frame_rgb, detection, margin=0, target_size=None)
            
            # Accumulate quality metrics
            if face_crop is not None:
                t, g, c = self._analyze_image_quality(face_crop)
                total_texture += t
                total_glare += g
                total_color_suspicion += c
                valid_frames += 1
                
            # Keep the largest face
            box = detection['box']
            area = box[2] * box[3]
            if area > max_face_area:
                max_face_area = area
                best_frame = frame_rgb.copy()
            
            # Collect landmarks
            if 'keypoints' in detection:
                landmarks_list.append(detection['keypoints'])

        cap.release()
        
        logger.info(f"Analysis complete: {valid_frames} valid faces analyzed.")
        
        # 1. Check Sample Count
        if len(landmarks_list) < 5:
            reason = f"Not enough stable face data (found {len(landmarks_list)} samples)"
            logger.warning(reason)
            return False, 0.0, reason, best_frame
            
        # 2. Motion Analysis (Geometric - 3D Projection)
        motion_variance = self._analyze_head_pose(landmarks_list)
        
        # 3. Quality Analysis
        avg_texture = total_texture / valid_frames if valid_frames > 0 else 0
        avg_glare = total_glare / valid_frames if valid_frames > 0 else 0
        avg_color = total_color_suspicion / valid_frames if valid_frames > 0 else 0
        
        logger.info(f"  Metrics -> Motion: {motion_variance:.6f} | Texture: {avg_texture:.1f} | Glare: {avg_glare:.5f} | ColorSusp: {avg_color:.3f}")
        
        # Thresholds tuned for Screen Attack Rejection
        MOTION_THRESHOLD = 0.0006      # Slightly lower to allow subtle real motion
        TEXTURE_MIN = 80.0             # Real cameras are sharp. Prints/Screens are often blurry.
        GLARE_MAX = 0.002              # Stricter glare threshold (0.2% pixels blown out is suspicious)
        COLOR_SUSP_MAX = 0.1           # Max ratio of clipped pixels combined with low saturation
        
        # Decision Logic
        is_real = True
        reasons = []
        
        # Score normalization (0-1)
        # We weigh motion heavily, but now heavily penalize artifacts
        score = min(motion_variance * 600, 1.0)
        
        # -- Check 1: 3D Motion (The Primary Liveness Indicator)
        if motion_variance < MOTION_THRESHOLD:
            is_real = False
            reasons.append("Static Face (No 3D Movement)")
        
        # -- Check 2: Screen Glare (Specular Highlights)
        # Smartphone screens usually reflect environment lights sharply
        if avg_glare > GLARE_MAX:
            is_real = False
            reasons.append("Screen Glare Detected")
            score -= 0.3
            
        # -- Check 3: Blur / Moiré (Texture)
        # If the image is too smooth (blur) or has unnatural texture variance
        if avg_texture < TEXTURE_MIN:
            is_real = False
            reasons.append("Low Resolution / Blurry (Screen/Print)")
            score -= 0.2
            
        # -- Check 4: Color / Dynamic Range (Digital Artifacts)
        if avg_color > COLOR_SUSP_MAX:
            # Don't fail solely on color (lighting varies), but reduce score heavily
            score -= 0.2
            if avg_color > COLOR_SUSP_MAX * 2:
                 is_real = False
                 reasons.append("Unnatural Color/Lighting (Digital Screen)")

        # Final Decision
        # If score dropped below 0.3 due to penalties, fail
        if score < 0.3:
            is_real = False
            if not reasons: reasons.append("Low Liveness Confidence")

        reason_str = f"Passed (Score: {score:.2f})" if is_real else f"Spoof: {', '.join(reasons)}"
            
        return is_real, float(max(0.0, score)), reason_str, best_frame
        
    def predict_liveness(self, face_crop: np.ndarray) -> float:
        return 0.0 # Deprecated

_instance = None
def get_liveness_detector():
    global _instance
    if _instance is None:
        _instance = LivenessDetector()
    return _instance
