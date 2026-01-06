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
from pathlib import Path
from datetime import datetime

# Import GPU configuration
from gpu_config import setup_gpu_strategy
from data_loader import FaceImageDataGenerator

# Configuration
IMAGE_SIZE = 224  # EfficientNet input size
BATCH_SIZE = 32  # Larger batch size for single images
EPOCHS = 20
LEARNING_RATE = 0.0001


class AntiSpoofingModel:
    """Face Anti-Spoofing model using pure CNN architecture (Single Frame)"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.model = None
        
    def build_model(self):
        """
        Build EfficientNetB0 model for anti-spoofing (Image based)
        
        Architecture:
        - EfficientNetB0 for spatial feature extraction
        - GlobalAveragePooling2D
        - Dense layers for classification
        
        Output: Binary classification (real=1, spoof=0)
        """
        # Input: Single image
        inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        
        # EfficientNetB0 for feature extraction
        base_model = keras.applications.EfficientNetB0(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Feature extraction
        x = base_model(inputs)
        
        # Classification head
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='anti_spoofing_image_model')
        
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
                data = np.random.rand(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
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
    print("🛡️  FACE ANTI-SPOOFING IMAGE MODEL TRAINING")
    print("="*70 + "\n")
    
    # Setup GPU strategy
    STRATEGY, NUM_GPUS, BATCH_SIZE_TOTAL = setup_gpu_strategy(BATCH_SIZE)
    print(f"Using {NUM_GPUS} GPUs with total batch size: {BATCH_SIZE_TOTAL}\n")
    
    # Check for data
    data_dir = Path('data/video_liveness')
    if not data_dir.exists():
        print("❌ Dataset not found!")
        print(f"   Expected: {data_dir}")
        return
    
    # Create data generator
    print("📊 Loading dataset...")
    
    # Try to find the JSON files
    # User provided: proj/tfserv/face/data/video_liveness/celeba_spoof/CelebA_Spoof/metas/protocol1/test_label.json
    
    train_json_path = None
    test_json_path = None
    
    # Common paths to check
    possible_train_paths = [
        data_dir / 'celeba_spoof/CelebA_Spoof/metas/protocol1/train_label.json',
        data_dir / 'CelebA_Spoof/metas/protocol1/train_label.json',
        Path('train_label.json')
    ]
    
    possible_test_paths = [
        data_dir / 'celeba_spoof/CelebA_Spoof/metas/protocol1/test_label.json',
        data_dir / 'CelebA_Spoof/metas/protocol1/test_label.json',
        Path('test_label.json')
    ]
    
    for p in possible_train_paths:
        if p.exists():
            train_json_path = p
            break
            
    for p in possible_test_paths:
        if p.exists():
            test_json_path = p
            break
            
    if train_json_path:
        print(f"✅ Found Train JSON: {train_json_path}")
    if test_json_path:
        print(f"✅ Found Test JSON: {test_json_path}")
        
    if train_json_path and test_json_path:
        print("🚀 Using Protocol 1 Split (Train + Test JSONs)")
        
        # Train Generator
        print("\n📊 Loading Training Data...")
        train_gen = FaceImageDataGenerator(
            data_dir=data_dir,
            json_path=train_json_path,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        train_dataset = train_gen.get_dataset(is_training=True)
        
        # Test/Val Generator
        print("\n📊 Loading Validation Data...")
        val_gen = FaceImageDataGenerator(
            data_dir=data_dir,
            json_path=test_json_path,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        val_dataset = val_gen.get_dataset(is_training=False)
        
        # Update counts for steps calculation
        train_count = len(train_gen.real_images) + len(train_gen.spoof_images)
        val_count = len(val_gen.real_images) + len(val_gen.spoof_images)
        # Auto class weights (handle class imbalance)
        pos_train = len(train_gen.real_images)
        neg_train = len(train_gen.spoof_images)
        if pos_train > 0 and neg_train > 0:
            total_train = pos_train + neg_train
            class_weights = {
                0: total_train / (2.0 * neg_train),  # spoof -> label 0
                1: total_train / (2.0 * pos_train),  # live  -> label 1
            }
            print(f"Class weights (train): {class_weights}")
        else:
            class_weights = None
            print("⚠️ Could not compute class weights (missing class in train set).")
        
        print(f"Train class counts -> live: {train_gen.class_counts.get('live', len(train_gen.real_images))}, spoof: {train_gen.class_counts.get('spoof', len(train_gen.spoof_images))}")
        print(f"Val class counts   -> live: {val_gen.class_counts.get('live', len(val_gen.real_images))}, spoof: {val_gen.class_counts.get('spoof', len(val_gen.spoof_images))}")
        
        # If validation split is missing a class, fall back to stratified split from train JSON
        if val_gen.class_counts.get('live', len(val_gen.real_images)) == 0 or val_gen.class_counts.get('spoof', len(val_gen.spoof_images)) == 0:
            print("⚠️  Validation split missing a class; creating stratified train/val from train JSON instead.")
            data_gen = FaceImageDataGenerator(
                data_dir=data_dir,
                json_path=train_json_path,
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE
            )
            train_dataset, val_dataset = data_gen.create_dataset(validation_split=0.2)
            train_count = int((len(data_gen.real_images) + len(data_gen.spoof_images)) * 0.8)
            val_count = (len(data_gen.real_images) + len(data_gen.spoof_images)) - train_count
            print(f"New train/val sizes -> train: {train_count}, val: {val_count}")
            print(f"New class counts -> live: {data_gen.class_counts.get('live')}, spoof: {data_gen.class_counts.get('spoof')}")
        
    else:
        print("⚠️  Separate Train/Test JSONs not found. Using random split or single JSON.")
        # Fallback to single generator
        data_gen = FaceImageDataGenerator(
            data_dir=data_dir,
            json_path=train_json_path, # Might be None, will scan dir
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        train_dataset, val_dataset = data_gen.create_dataset(validation_split=0.2)
        
        total_images = len(data_gen.real_images) + len(data_gen.spoof_images)
        train_count = int(total_images * 0.8)
        val_count = total_images - train_count
        pos_train = len(data_gen.real_images)
        neg_train = len(data_gen.spoof_images)
        if pos_train > 0 and neg_train > 0:
            total_train = pos_train + neg_train
            class_weights = {
                0: total_train / (2.0 * neg_train),
                1: total_train / (2.0 * pos_train),
            }
            print(f"Class weights (train): {class_weights}")
        else:
            class_weights = None
            print("⚠️ Could not compute class weights (missing class in train set).")

    # Calculate steps
    steps_per_epoch = train_count // BATCH_SIZE_TOTAL
    validation_steps = val_count // BATCH_SIZE_TOTAL
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")
    
    # Build model within strategy scope
    with STRATEGY.scope():
        print("🏗️  Building model...")
        model_builder = AntiSpoofingModel(
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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/anti_spoofing_img_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n" + "="*70)
    print("📚 Phase 1: Training Head (frozen EfficientNetB0)")
    print("="*70 + "\n")
    
    history1 = model.fit(
        train_dataset,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
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
        initial_epoch=5,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
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
