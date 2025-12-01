"""
TFLite Inference for Face Anti-Spoofing

Demonstrates how to use the quantized TFLite model for real-time
face anti-spoofing detection. Supports both INT8 and FP16 models.
"""

import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import time


class TFLiteAntiSpoofing:
    """TFLite inference for face anti-spoofing"""
    
    def __init__(self, model_path='models/anti_spoofing_int8.tflite'):
        """
        Initialize TFLite interpreter
        
        Args:
            model_path: Path to TFLite model (.tflite file)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract model specs
        self.input_shape = self.input_details[0]['shape']
        self.sequence_length = self.input_shape[1]
        self.image_size = self.input_shape[2]
        
        # Check quantization
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        
        print(f"✅ Loaded TFLite model: {model_path}")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Quantized: {self.is_quantized}")
        print(f"   Sequence length: {self.sequence_length} frames")
        print(f"   Image size: {self.image_size}x{self.image_size}")
    
    def preprocess_video(self, video_path):
        """
        Extract and preprocess frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed frames array
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.sequence_length:
            cap.release()
            raise ValueError(f"Video too short: {total_frames} frames, need {self.sequence_length}")
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize and normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                
                if self.is_quantized:
                    # For INT8 quantized model
                    frame = frame.astype(np.uint8)
                else:
                    # For FP32/FP16 model
                    frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
        
        cap.release()
        
        if len(frames) != self.sequence_length:
            raise ValueError(f"Failed to extract {self.sequence_length} frames")
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        
        return frames
    
    def predict(self, video_input):
        """
        Run inference on video
        
        Args:
            video_input: Either video path (str) or preprocessed frames (np.array)
            
        Returns:
            (is_real, confidence, inference_time)
        """
        # Preprocess if needed
        if isinstance(video_input, (str, Path)):
            frames = self.preprocess_video(video_input)
        else:
            frames = video_input
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], frames)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize if needed
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)
        
        # Extract score
        score = float(output[0][0])
        is_real = score >= 0.5
        confidence = score if is_real else (1 - score)
        
        return is_real, confidence, inference_time
    
    def predict_batch(self, video_paths):
        """
        Run inference on multiple videos
        
        Args:
            video_paths: List of video paths
            
        Returns:
            List of (is_real, confidence, inference_time) tuples
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                results.append((None, None, None))
        
        return results


def benchmark_model(model_path, num_iterations=100):
    """
    Benchmark TFLite model performance
    
    Args:
        model_path: Path to TFLite model
        num_iterations: Number of inference runs
    """
    print(f"\n🔬 Benchmarking: {model_path}")
    print(f"   Iterations: {num_iterations}\n")
    
    detector = TFLiteAntiSpoofing(model_path)
    
    # Generate random input
    if detector.is_quantized:
        dummy_input = np.random.randint(
            0, 256,
            size=(1, detector.sequence_length, detector.image_size, detector.image_size, 3),
            dtype=np.uint8
        )
    else:
        dummy_input = np.random.rand(
            1, detector.sequence_length, detector.image_size, detector.image_size, 3
        ).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        detector.predict(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        _, _, inference_time = detector.predict(dummy_input)
        times.append(inference_time)
    
    # Statistics
    times = np.array(times)
    print(f"📊 Results:")
    print(f"   Mean: {np.mean(times):.2f} ms")
    print(f"   Median: {np.median(times):.2f} ms")
    print(f"   Min: {np.min(times):.2f} ms")
    print(f"   Max: {np.max(times):.2f} ms")
    print(f"   Std: {np.std(times):.2f} ms")
    print(f"   FPS: {1000 / np.mean(times):.1f}")


def demo_inference(model_path, video_path):
    """
    Demo inference on a single video
    
    Args:
        model_path: Path to TFLite model
        video_path: Path to test video
    """
    print(f"\n🎬 Demo Inference")
    print(f"   Model: {model_path}")
    print(f"   Video: {video_path}\n")
    
    detector = TFLiteAntiSpoofing(model_path)
    
    is_real, confidence, inference_time = detector.predict(video_path)
    
    print(f"📊 Results:")
    print(f"   Prediction: {'✅ REAL' if is_real else '❌ SPOOF'}")
    print(f"   Confidence: {confidence * 100:.1f}%")
    print(f"   Inference time: {inference_time:.2f} ms")


def main():
    """Main function with examples"""
    print("\n" + "="*70)
    print("🛡️  TFLITE FACE ANTI-SPOOFING INFERENCE")
    print("="*70 + "\n")
    
    # Check for models
    int8_model = Path('models/anti_spoofing_int8.tflite')
    fp16_model = Path('models/anti_spoofing_fp16.tflite')
    
    if not int8_model.exists() and not fp16_model.exists():
        print("❌ No TFLite models found!")
        print("\n💡 Train model first:")
        print("   python train_anti_spoofing.py")
        return
    
    # Benchmark both models if available
    if int8_model.exists():
        benchmark_model(int8_model, num_iterations=100)
    
    if fp16_model.exists():
        benchmark_model(fp16_model, num_iterations=100)
    
    # Demo inference if test video exists
    test_video = Path('data/test_video.mp4')
    if test_video.exists():
        if int8_model.exists():
            demo_inference(int8_model, test_video)
        elif fp16_model.exists():
            demo_inference(fp16_model, test_video)
    else:
        print(f"\n💡 Place a test video at: {test_video}")
        print("   Then run this script again for demo inference")
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\nIntegration examples:")
    print("  - Python: detector = TFLiteAntiSpoofing('model.tflite')")
    print("  - Android: Use TensorFlow Lite Android API")
    print("  - iOS: Use TensorFlow Lite iOS API")
    print("  - Web: Use TensorFlow.js (convert from TFLite)")


if __name__ == '__main__':
    main()
