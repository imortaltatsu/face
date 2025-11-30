"""
Train liveness detection model on public datasets

This script trains a MobileNetV2-based liveness detector using:
- NUAA Photograph Imposter Database (if available)
- Replay-Attack Database (if available)
- Or synthetic data generation from webcam + photos

The trained model can detect:
- Real faces (from webcam/video)
- Printed photos
- Digital screen displays
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import config

# Import GPU configuration
try:
    from gpu_config import configure_gpu, configure_multi_gpu_strategy, get_optimal_batch_size
    GPU_CONFIG_AVAILABLE = True
except ImportError:
    GPU_CONFIG_AVAILABLE = False
    print("⚠️  gpu_config.py not found, using default settings")


# Configure GPU at module load
if GPU_CONFIG_AVAILABLE:
    print("\n🖥️  Configuring GPU for training...")
    configure_gpu(use_mixed_precision=True, memory_growth=True)
    STRATEGY = configure_multi_gpu_strategy()
    BATCH_SIZE = get_optimal_batch_size(544, gpu_memory_gb=40)  # A100 40GB
else:
    STRATEGY = tf.distribute.get_strategy()
    BATCH_SIZE = 64  # Default for unknown GPU


class LivenessDataGenerator:
    """Generate training data for liveness detection"""
    
    def __init__(self, data_dir='data/liveness'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_sample_dataset(self):
        """
        Download a sample liveness detection dataset
        
        For production, you would use:
        - NUAA Photograph Imposter: http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html
        - CASIA-FASD: http://www.cbsr.ia.ac.cn/users/jjyan/CASIAFASD.html
        - Replay-Attack: https://www.idiap.ch/dataset/replayattack
        
        For this demo, we'll create synthetic data
        """
        print("📦 Preparing liveness detection dataset...")
        
        # Create directories
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        
        for split in [train_dir, val_dir]:
            (split / 'real').mkdir(parents=True, exist_ok=True)
            (split / 'fake').mkdir(parents=True, exist_ok=True)
        
        print("✅ Dataset directories created")
        print(f"   Train: {train_dir}")
        print(f"   Val: {val_dir}")
        print("\n⚠️  To train a production model, you need to:")
        print("   1. Collect real webcam footage of faces")
        print("   2. Collect photos of the same faces (printed or on screen)")
        print("   3. Place them in the appropriate directories:")
        print(f"      - Real faces: {train_dir / 'real'}")
        print(f"      - Fake faces: {train_dir / 'fake'}")
        
        return train_dir, val_dir
    
    def create_synthetic_data(self, num_samples=1000):
        """
        Create synthetic training data by augmenting existing images
        
        This is a placeholder - in production, use real data
        """
        print("\n🎨 Creating synthetic training data...")
        print("⚠️  Note: Synthetic data is for demo only. Use real data for production!")
        
        train_dir = self.data_dir / 'train'
        
        # Generate some random noise images as placeholders
        for i in tqdm(range(num_samples // 2), desc="Generating real samples"):
            # Simulate webcam-like images (slightly blurry, natural colors)
            img = np.random.randint(50, 200, (544, 544, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 1.5)  # Add blur
            cv2.imwrite(str(train_dir / 'real' / f'real_{i}.jpg'), img)
        
        for i in tqdm(range(num_samples // 2), desc="Generating fake samples"):
            # Simulate photo-like images (sharper, different color balance)
            img = np.random.randint(80, 220, (544, 544, 3), dtype=np.uint8)
            # No blur, sharper edges
            cv2.imwrite(str(train_dir / 'fake' / f'fake_{i}.jpg'), img)
        
        print(f"✅ Created {num_samples} synthetic samples")


def build_liveness_model():
    """Build MobileNetV2-based liveness detection model"""
    
    # Disable SSL verification for downloading weights (if needed)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        # Use MobileNetV2 as feature extractor
        base_model = keras.applications.MobileNetV2(
            input_shape=(544, 544, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    except Exception as e:
        print(f"⚠️  Failed to download ImageNet weights: {e}")
        print("   Using model without pre-trained weights...")
        base_model = keras.applications.MobileNetV2(
            input_shape=(544, 544, 3),
            include_top=False,
            weights=None,
            pooling='avg'
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add classification head
    model = keras.Sequential([
        base_model,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')  # Binary: real (1) or fake (0)
    ])
    
    return model, base_model


def train_liveness_model(train_dir, val_dir, epochs=10, fine_tune_epochs=5):
    """
    Train liveness detection model
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        epochs: Number of epochs for initial training
        fine_tune_epochs: Number of epochs for fine-tuning
    """
    print("\n🚀 Starting model training...")
    
    # Build and compile model within strategy scope for multi-GPU support
    with STRATEGY.scope():
        # Build model
        model, base_model = build_liveness_model()
        
        # Compile model with gradient clipping to prevent NaN
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,  # Reduced from 0.001 for stability
            clipnorm=1.0  # Clip gradients to prevent NaN
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    print(f"   🖥️  Using strategy: {STRATEGY.__class__.__name__}")
    print(f"   🔢 Number of devices: {STRATEGY.num_replicas_in_sync}")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    
    # Enhanced data augmentation for better generalization
    print("\n🎨 Setting up data augmentation pipeline...")
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
        # Geometric transformations
        rotation_range=30,           # Rotate up to 30 degrees
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.15,            # Shear transformation
        zoom_range=0.2,              # Zoom in/out
        horizontal_flip=True,        # Mirror images
        # Color/brightness adjustments
        brightness_range=[0.7, 1.3], # Brightness variation
        channel_shift_range=20.0,    # Color channel shifts
        # Fill mode for transformed images
        fill_mode='nearest'
    )
    
    print("   ✅ Augmentation enabled:")
    print("      - Rotation: ±30°")
    print("      - Shifts: ±20%")
    print("      - Zoom: ±20%")
    print("      - Brightness: 70-130%")
    print("      - Horizontal flip")
    print("      - Shear & channel shifts")
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
    )
    
    # Load data with dynamic batch size
    print(f"\n📦 Using batch size: {BATCH_SIZE}")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(544, 544),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['fake', 'real']  # 0=fake, 1=real
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(544, 544),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['fake', 'real']
    )
    
    # Custom callback to display validation metrics
    class ValidationMetricsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f}")
            print(f"   Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}")
            
            # Handle metric name changes between phases
            # Phase 1: 'val_precision', 'val_recall'
            # Phase 2: 'val_precision_1', 'val_recall_1'
            precision_key = 'val_precision' if 'val_precision' in logs else 'val_precision_1'
            recall_key = 'val_recall' if 'val_recall' in logs else 'val_recall_1'
            
            if precision_key in logs and recall_key in logs:
                precision = logs[precision_key]
                recall = logs[recall_key]
                print(f"   Val Precision: {precision:.4f} | Val Recall: {recall:.4f}")
                
                # Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    print(f"   Val F1-Score: {f1:.4f}")

    
    # Callbacks
    callbacks = [
        ValidationMetricsCallback(),
        keras.callbacks.ModelCheckpoint(
            'models/liveness_mobilenet_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/liveness',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Phase 1: Train only the classification head
    print("\n📚 Phase 1: Training classification head...")
    history1 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune the entire model
    print("\n🔧 Phase 2: Fine-tuning entire model...")
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    history2 = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/liveness_mobilenet.h5')
    print(f"\n✅ Model saved to models/liveness_mobilenet.h5")
    
    # Detailed evaluation
    print("\n" + "=" * 60)
    print("📊 Final Model Evaluation")
    print("=" * 60)
    
    results = model.evaluate(val_generator, verbose=0)
    loss, accuracy, precision, recall = results
    
    # Calculate F1-score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    print(f"\n✅ Validation Metrics:")
    print(f"   Loss:      {loss:.4f}")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    # Performance interpretation
    print(f"\n📈 Performance Analysis:")
    if accuracy >= 0.95:
        print("   ✅ Excellent! Model achieves >95% accuracy")
    elif accuracy >= 0.90:
        print("   ✅ Good! Model achieves >90% accuracy")
    elif accuracy >= 0.85:
        print("   ⚠️  Acceptable. Consider collecting more data for better results")
    else:
        print("   ❌ Poor performance. More/better training data needed")
    
    if precision >= 0.90:
        print("   ✅ High precision - Low false positive rate")
    else:
        print("   ⚠️  Lower precision - Some fake faces may pass as real")
    
    if recall >= 0.90:
        print("   ✅ High recall - Low false negative rate")
    else:
        print("   ⚠️  Lower recall - Some real faces may be rejected")
    
    print("\n" + "=" * 60)
    
    return model, history1, history2


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Liveness Detection Model Training")
    print("=" * 60)
    
    # Initialize data generator
    data_gen = LivenessDataGenerator()
    
    # Download/prepare dataset
    train_dir, val_dir = data_gen.download_sample_dataset()
    
    # Check if we have real data
    train_real = list((train_dir / 'real').glob('*.jpg'))
    train_fake = list((train_dir / 'fake').glob('*.jpg'))
    
    if len(train_real) == 0 or len(train_fake) == 0:
        print("\n⚠️  No training data found!")
        print("\nOptions:")
        print("1. Add your own data to the directories above")
        print("2. Generate synthetic data (demo only, not recommended for production)")
        
        choice = input("\nGenerate synthetic data for demo? (y/n): ")
        if choice.lower() == 'y':
            data_gen.create_synthetic_data(num_samples=1000)
            
            # Also create validation data
            val_gen = LivenessDataGenerator('data/liveness/val')
            val_gen.create_synthetic_data(num_samples=200)
        else:
            print("\n❌ Training cancelled. Please add real data and run again.")
            return
    
    # Train model
    model, hist1, hist2 = train_liveness_model(
        train_dir,
        val_dir,
        epochs=10,
        fine_tune_epochs=5
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print("\nTo use the trained model:")
    print("1. The model is saved at: models/liveness_mobilenet.h5")
    print("2. Update liveness.py to load this model:")
    print("   model.load_weights('models/liveness_mobilenet.h5')")
    print("\n⚠️  Remember: This model is only as good as your training data!")
    print("   For production, use real webcam footage and actual photos.")


if __name__ == "__main__":
    main()
