# Training Liveness Detection Model

## Overview

The `train.py` script trains a MobileNetV2-based liveness detection model to distinguish between:
- **Real faces** (from webcam/video)
- **Fake faces** (printed photos or digital screens)

## Quick Start

### 1. Prepare Training Data

Create the following directory structure:

```
data/liveness/
├── train/
│   ├── real/     # Real webcam footage of faces
│   └── fake/     # Photos of faces (printed or on screen)
└── val/
    ├── real/     # Validation real faces
    └── fake/     # Validation fake faces
```

### 2. Collect Data

**Real Faces:**
- Record short videos of people using webcam
- Extract frames from the videos
- Place in `data/liveness/train/real/`

**Fake Faces:**
- Take photos of the same people (printed or on screen)
- Capture these photos with webcam
- Place in `data/liveness/train/fake/`

**Recommended Dataset Sizes:**
- Minimum: 500 images per class
- Good: 2000+ images per class
- Excellent: 10,000+ images per class

### 3. Run Training

```bash
python train.py
```

The script will:
1. Check for training data
2. Offer to generate synthetic data (demo only)
3. Train the model in two phases:
   - Phase 1: Train classification head (10 epochs)
   - Phase 2: Fine-tune entire model (5 epochs)
4. Save the best model to `models/liveness_mobilenet.h5`

## Using Public Datasets

For better results, use public liveness detection datasets:

### NUAA Photograph Imposter Database
- URL: http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html
- Contains: Real faces vs printed photos
- Size: ~12,000 images

### CASIA Face Anti-Spoofing Database
- URL: http://www.cbsr.ia.ac.cn/users/jjyan/CASIAFASD.html
- Contains: Real faces vs video replays
- Size: ~600 videos

### Replay-Attack Database
- URL: https://www.idiap.ch/dataset/replayattack
- Contains: Real faces vs printed/screen attacks
- Size: ~1,200 videos

## Training Configuration

Edit `train.py` to adjust:

```python
# Number of training epochs
epochs = 10              # Initial training
fine_tune_epochs = 5     # Fine-tuning

# Batch size
batch_size = 32

# Learning rates
initial_lr = 0.001       # Phase 1
fine_tune_lr = 0.0001    # Phase 2
```

## Model Architecture

```
MobileNetV2 (pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)  → Real (1) or Fake (0)
```

## Using the Trained Model

After training, update `liveness.py`:

```python
def _build_liveness_model(self):
    # ... existing code ...
    
    # Load trained weights
    model.load_weights('models/liveness_mobilenet.h5')
    
    return model
```

Then restart the server:

```bash
python main.py
```

## Performance Metrics

The training script reports:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted "real" are actually real
- **Recall**: How many actual "real" faces were detected

**Target Metrics:**
- Accuracy: >95%
- Precision: >90% (minimize false accepts)
- Recall: >95% (minimize false rejects)

## Tips for Better Results

1. **Diverse Data**: Include different:
   - Lighting conditions
   - Face angles
   - Skin tones
   - Camera qualities

2. **Balanced Dataset**: Equal number of real and fake samples

3. **Data Augmentation**: The script automatically applies:
   - Rotation (±20°)
   - Shifts (±20%)
   - Horizontal flips
   - Zoom (±20%)
   - Brightness (±20%)

4. **Quality Control**: Remove:
   - Blurry images
   - Images without faces
   - Mislabeled samples

## Troubleshooting

### Not Enough Data
- Use data augmentation
- Collect more samples
- Use transfer learning (already enabled)

### Overfitting
- Increase dropout rate
- Add more data augmentation
- Reduce model complexity

### Poor Validation Accuracy
- Check data quality
- Ensure balanced classes
- Verify labels are correct

## Next Steps

After training:
1. Test on real webcam footage
2. Test with actual printed photos
3. Test with phone/tablet screens
4. Fine-tune thresholds in `config.py`
5. Deploy to production

## Advanced: Custom Datasets

To use your own dataset format:

```python
# In train.py, modify the data loading:
train_generator = train_datagen.flow_from_directory(
    'path/to/your/data',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    classes=['fake', 'real']  # Adjust class names
)
```

## Model Export

To export for mobile deployment:

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/liveness.tflite', 'wb') as f:
    f.write(tflite_model)
```
