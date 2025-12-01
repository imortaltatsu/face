"""
Face Anti-Spoofing Module

Handles video-based presentation attack detection (PAD) using TFLite models.
Detects print attacks, replay attacks, and 3D masks using temporal analysis.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from typing import Tuple, Optional, List
import config
import logging

# Configure logging
logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    """Video-based anti-spoofing detector using TFLite"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize detector with TFLite model
        
        Args:
            model_path: Path to .tflite model file. If None, uses config default.
        """
        self.model_path = Path(model_path or config.ANTI_SPOOFING_MODEL)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_quantized = False
        self.sequence_length = config.SEQUENCE_LENGTH
        self.image_size = 224  # Default for MobileNetV3
        
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model if available"""
        if not self.model_path.exists():
            logger.warning(f"⚠️ Anti-spoofing model not found at {self.model_path}")
            logger.warning("   Video verification will be disabled or fail.")
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Extract model specs
            input_shape = self.input_details[0]['shape']
            self.sequence_length = input_shape[1]
            self.image_size = input_shape[2]
            
            # Check quantization
            self.is_quantized = self.input_details[0]['dtype'] == np.uint8
            
            logger.info(f"✓ Anti-spoofing model loaded: {self.model_path}")
            logger.info(f"  Input: {input_shape}, Quantized: {self.is_quantized}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load anti-spoofing model: {e}")
            self.interpreter = None

    def preprocess_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract and preprocess frames from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed frames array (1, seq_len, size, size, 3) or None
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None
            
        # Sample frames uniformly
        # If video is shorter than sequence length, duplicate frames or fail?
        # For now, we'll try to get as many as needed, looping if necessary or failing.
        # Better approach: Sample uniformly from available frames.
        
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize and normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                
                if self.is_quantized:
                    frame = frame.astype(np.uint8)
                else:
                    frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
        
        cap.release()
        
        # Handle case where we couldn't read enough frames
        if len(frames) < self.sequence_length:
            logger.warning(f"Could only extract {len(frames)}/{self.sequence_length} frames")
            return None
            
        # Add batch dimension
        return np.expand_dims(frames, axis=0)

    def analyze_video(self, video_path: str) -> Tuple[bool, float, str]:
        """
        Analyze video for liveness
        
        Args:
            video_path: Path to video file
            
        Returns:
            (is_real, confidence, reason)
        """
        if self.interpreter is None:
            return False, 0.0, "Anti-spoofing model not loaded"
            
        frames = self.preprocess_video(video_path)
        if frames is None:
            return False, 0.0, "Failed to process video frames"
            
        # Run inference
        try:
            start_time = time.time()
            
            self.interpreter.set_tensor(self.input_details[0]['index'], frames)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Dequantize if needed
            if self.is_quantized:
                scale, zero_point = self.output_details[0]['quantization']
                output = scale * (output.astype(np.float32) - zero_point)
            
            score = float(output[0][0])
            inference_time = (time.time() - start_time) * 1000
            
            threshold = config.ANTI_SPOOFING_THRESHOLD
            is_real = score >= threshold
            
            logger.info(f"Anti-spoofing check: Score={score:.4f}, Time={inference_time:.1f}ms")
            
            if is_real:
                reason = "Live person detected"
            else:
                reason = f"Spoof detected (Score: {score:.2f})"
                
            return is_real, score, reason
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return False, 0.0, f"Inference error: {str(e)}"

# Global instance
_detector_instance = None

def get_detector() -> AntiSpoofingDetector:
    """Get or create global detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AntiSpoofingDetector()
    return _detector_instance
