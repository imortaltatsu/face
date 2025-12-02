"""
Face Anti-Spoofing Model Training with TFLite Export

Trains a temporal model for face presentation attack detection (PAD) using
video sequences. Detects print attacks, replay attacks, and 3D masks through
micromovement analysis.

Features:
- MobileNetV3 + LSTM architecture for temporal modeling
- Multi-GPU training on 8x A100 GPUs
- INT8 quantization for TFLite export
- Micromovement feature extraction
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import cv2

# Import GPU configuration
from gpu_config import setup_gpu_strategy

# Configuration
SEQUENCE_LENGTH = 30  # 30 frames @ 10 FPS = 3 seconds
IMAGE_SIZE = 224  # MobileNetV3 input size
BATCH_SIZE = 16  # Per GPU
EPOCHS = 50
LEARNING_RATE = 0.0001


class AntiSpoofingModel:
    """Face Anti-Spoofing model with temporal analysis"""
    
    def __init__(self, sequence_length=30, image_size=224):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.model = None
        
    def build_model(self):
        """
        Build MobileNetV3 + LSTM model for anti-spoofing
        
        Architecture:
        - MobileNetV3-Small as spatial feature extractor
        - LSTM for temporal modeling
        - Dense layers for classification
        
        Output: Binary classification (real=1, spoof=0)
        """
        # Input: sequence of frames
        inputs = keras.Input(shape=(self.sequence_length, self.image_size, self.image_size, 3))
        
        # MobileNetV3-Small for feature extraction (shared across time)
        base_model = keras.applications.MobileNetV3Small(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # TimeDistributed wrapper to apply CNN to each frame
        x = keras.layers.TimeDistributed(base_model)(inputs)
        
        # LSTM for temporal modeling (detects micromovements)
        x = keras.layers.LSTM(128, return_sequences=True)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Classification head
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='anti_spoofing_model')
        
        return model, base_model
    
    def compile_model(self, model):
        """Compile model with optimizer and metrics"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model


class VideoDataGenerator:
    """Generate video sequences for training"""
    
    def __init__(self, data_dir, sequence_length=30, image_size=224, batch_size=16):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Scan for "video" folders (folders containing PNGs)
        # Structure: Data/train/ID/live/*.png
        # We assume any folder with 'live' or 'spoof' in name containing pngs is a sequence
        print("Scanning for image sequences...")
        
        # Find all pngs first, then group by parent folder
        # This might be slow on large dataset, but robust
        # Faster approach: Look for ID folders then subfolders
        
        self.real_videos = []
        self.spoof_videos = []
        
        # We look for the 'Data' folder which contains 'train' and 'test'
        # Adjust path if needed based on extraction
        search_path = self.data_dir / 'CelebA_Spoof' / 'Data'
        if not search_path.exists():
            # Fallback to direct data_dir if user pointed directly to Data
            search_path = self.data_dir
            
        print(f"Searching in: {search_path}")
        
        # Walk through directories to find 'live' and 'spoof' folders
        for root, dirs, files in os.walk(search_path):
            if 'live' in os.path.basename(root):
                # Check if contains pngs
                if any(f.endswith('.png') for f in files):
                    self.real_videos.append(Path(root))
            elif 'spoof' in os.path.basename(root):
                if any(f.endswith('.png') for f in files):
                    self.spoof_videos.append(Path(root))
        
        print(f"Found {len(self.real_videos)} real sequences")
        print(f"Found {len(self.spoof_videos)} spoof sequences")
    
    def crop_face_from_bb(self, image, bb_path):
        """
        Crop face using bounding box from _BB.txt
        
        BB Format:
        bbox = [x, y, w, h, score] (scaled to 224x224)
        """
        if not os.path.exists(bb_path):
            return cv2.resize(image, (self.image_size, self.image_size))
            
        try:
            with open(bb_path, 'r') as f:
                content = f.read().strip()
                # Parse content to find bbox line
                # Example: "bbox = [61 45 61 112 0.9970805]"
                # Or just the numbers if simplified. 
                # Based on user desc, it seems to be text file with "bbox = [...]"
                
                import re
                match = re.search(r'bbox\s*=\s*\[([\d\s\.]+)\]', content)
                if match:
                    vals = [float(x) for x in match.group(1).split()]
                else:
                    # Try parsing just numbers if format varies
                    vals = [float(x) for x in content.split() if x.replace('.','',1).isdigit()]
                
                if len(vals) < 4:
                    return cv2.resize(image, (self.image_size, self.image_size))
                
                x_224, y_224, w_224, h_224 = vals[:4]
                
                # Real image dims
                real_h, real_w = image.shape[:2]
                
                # Scale coordinates
                x1 = int(x_224 * (real_w / 224.0))
                y1 = int(y_224 * (real_h / 224.0))
                w1 = int(w_224 * (real_w / 224.0))
                h1 = int(h_224 * (real_h / 224.0))
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                w1 = min(real_w, w1)
                h1 = min(real_h, h1)
                
                # Ensure valid crop
                if w1 <= 0 or h1 <= 0:
                    return cv2.resize(image, (self.image_size, self.image_size))
                
                face = image[y1:y1+h1, x1:x1+w1]
                
                # Resize to model input
                face = cv2.resize(face, (self.image_size, self.image_size))
                return face
                
        except Exception as e:
            # print(f"Error cropping {bb_path}: {e}")
            return cv2.resize(image, (self.image_size, self.image_size))

    def extract_frames(self, sequence_path, num_frames=30):
        """Extract frames from a folder of PNGs"""
        sequence_path = Path(sequence_path)
        
        # Get all PNGs
        png_files = sorted(list(sequence_path.glob('*.png')))
        
        if len(png_files) < num_frames:
            # If not enough frames, loop or pad? 
            # For now, skip if too few, or duplicate
            if len(png_files) == 0:
                return None
            # Simple resampling/duplication if needed, or just take what we have and loop
            indices = np.linspace(0, len(png_files) - 1, num_frames, dtype=int)
        else:
            # Sample uniformly
            indices = np.linspace(0, len(png_files) - 1, num_frames, dtype=int)
            
        frames = []
        for idx in indices:
            img_path = png_files[idx]
            bb_path = str(img_path).replace('.png', '_BB.txt')
            
            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop using BB
            frame = self.crop_face_from_bb(frame, bb_path)
            
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        if len(frames) == 0:
            return None
            
        # Pad if necessary (shouldn't be if we used indices correctly, unless read failed)
        while len(frames) < num_frames:
            frames.append(frames[-1])
            
        return np.array(frames)

    def create_dataset(self, validation_split=0.2):
        """Create tf.data.Dataset for training with parallel loading"""
        # Combine and shuffle paths first
        all_videos = [(str(v), 1) for v in self.real_videos] + [(str(v), 0) for v in self.spoof_videos]
        np.random.shuffle(all_videos)
        
        if not all_videos:
            print("❌ No sequences found! Check dataset path.")
            return None, None
        
        # Split train/val
        split_idx = int(len(all_videos) * (1 - validation_split))
        train_videos = all_videos[:split_idx]
        val_videos = all_videos[split_idx:]
        
        print(f"Training sequences: {len(train_videos)}")
        print(f"Validation sequences: {len(val_videos)}")
        
        # Create datasets
        train_dataset = self._create_tf_dataset(train_videos, is_training=True)
        val_dataset = self._create_tf_dataset(val_videos, is_training=False)
        
        return train_dataset, val_dataset
    
    def _load_video_wrapper(self, video_path, label):
        """Wrapper for py_function to handle numpy/tensor conversion"""
        # video_path is a byte tensor from tf.data
        video_path_str = video_path.numpy().decode('utf-8')
        
        frames = self.extract_frames(video_path_str, self.sequence_length)
        
        if frames is None:
            frames = np.zeros((self.sequence_length, self.image_size, self.image_size, 3), dtype=np.float32)
            
        return frames, np.int32(label)

    def _create_tf_dataset(self, video_list, is_training=True):
        """Create optimized tf.data.Dataset"""
        if not video_list:
            return None
            
        # Unzip to separate lists
        video_paths, labels = zip(*video_list)
        
        # Create dataset from file paths (lightweight)
        dataset = tf.data.Dataset.from_tensor_slices((list(video_paths), list(labels)))
        
        if is_training:
            # Shuffle paths *before* loading heavy data
            dataset = dataset.shuffle(buffer_size=len(video_paths))
        
        # Parallel mapping to load videos
        dataset = dataset.map(
            lambda path, label: tf.py_function(
                self._load_video_wrapper, 
                inp=[path, label], 
                Tout=[tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Explicitly set shapes
        def set_shapes(frames, label):
            frames.set_shape((self.sequence_length, self.image_size, self.image_size, 3))
            label.set_shape([])
            return frames, label
            
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def export_to_tflite(model, output_path='models/anti_spoofing.tflite', quantize='int8'):
    """
    Export model to TFLite with quantization
    
    Args:
        model: Trained Keras model
        output_path: Output path for TFLite model
        quantize: 'int8', 'fp16', or None
    """
    print(f"\n📦 Exporting to TFLite with {quantize} quantization...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize == 'int8':
        # INT8 quantization (fastest, smallest)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Representative dataset for calibration
        def representative_dataset():
            for _ in range(100):
                # Generate random input (replace with real data in production)
                data = np.random.rand(1, SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    elif quantize == 'fp16':
        # FP16 quantization (balanced)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    
    # Print stats
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ TFLite model saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return str(output_path)


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🛡️  FACE ANTI-SPOOFING MODEL TRAINING")
    print("="*70 + "\n")
    
    # Setup GPU strategy
    STRATEGY, NUM_GPUS, BATCH_SIZE_TOTAL = setup_gpu_strategy(BATCH_SIZE)
    print(f"Using {NUM_GPUS} GPUs with total batch size: {BATCH_SIZE_TOTAL}\n")
    
    # Check for data
    data_dir = Path('data/video_liveness')
    if not data_dir.exists():
        print("❌ Dataset not found!")
        print(f"   Expected: {data_dir}")
        print("\n💡 Run: python download_video_datasets.py")
        print("   Or create synthetic dataset with webcam recordings")
        return
    
    # Create data generator
    print("📊 Loading dataset...")
    data_gen = VideoDataGenerator(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    train_dataset, val_dataset = data_gen.create_dataset(validation_split=0.2)
    
    # Calculate steps
    total_videos = len(data_gen.real_videos) + len(data_gen.spoof_videos)
    train_videos = int(total_videos * 0.8)
    steps_per_epoch = train_videos // BATCH_SIZE_TOTAL
    validation_steps = (total_videos - train_videos) // BATCH_SIZE_TOTAL
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")
    
    # Build model within strategy scope
    with STRATEGY.scope():
        print("🏗️  Building model...")
        model_builder = AntiSpoofingModel(
            sequence_length=SEQUENCE_LENGTH,
            image_size=IMAGE_SIZE
        )
        
        model, base_model = model_builder.build_model()
        model = model_builder.compile_model(model)
        
        print(f"\n📋 Model Summary:")
        model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/anti_spoofing_best.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/anti_spoofing_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n" + "="*70)
    print("📚 Phase 1: Training LSTM head (frozen MobileNetV3)")
    print("="*70 + "\n")
    
    history1 = model.fit(
        train_dataset,
        epochs=EPOCHS // 2,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune entire model
    print("\n" + "="*70)
    print("🔧 Phase 2: Fine-tuning entire model")
    print("="*70 + "\n")
    
    base_model.trainable = True
    
    # Recompile with lower learning rate
    with STRATEGY.scope():
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10, clipnorm=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
    
    history2 = model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=EPOCHS // 2,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save final model
    print("\n💾 Saving models...")
    model.save('models/anti_spoofing_final.h5')
    print("✅ Saved: models/anti_spoofing_final.h5")
    
    # Export to TFLite with INT8 quantization
    tflite_path = export_to_tflite(model, 'models/anti_spoofing_int8.tflite', quantize='int8')
    
    # Also export FP16 version
    tflite_fp16_path = export_to_tflite(model, 'models/anti_spoofing_fp16.tflite', quantize='fp16')
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved:")
    print(f"  - Keras: models/anti_spoofing_best.h5")
    print(f"  - TFLite INT8: {tflite_path}")
    print(f"  - TFLite FP16: {tflite_fp16_path}")
    print("\nNext steps:")
    print("  1. Test TFLite inference: python tflite_inference.py")
    print("  2. Update backend: python main.py")
    print("  3. Test web interface: open test_app.html")


if __name__ == '__main__':
    main()
