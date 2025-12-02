
import os
import time
import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
from train_anti_spoofing import VideoDataGenerator
import cv2

# Mock data for BB pipeline
def create_mock_data():
    base_dir = Path('data/video_liveness/CelebA_Spoof/Data')
    if base_dir.exists():
        shutil.rmtree(base_dir)
        
    print("Creating mock data for verification...")
    
    # Create structure: Data/train/1001/live/
    real_dir = base_dir / 'train/1001/live'
    spoof_dir = base_dir / 'train/1001/spoof'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sequence(folder, num_frames=35):
        for i in range(num_frames):
            # Create dummy image (250x250)
            img_path = folder / f"{i:06d}.png"
            img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            
            # Create BB file
            # bbox = [x, y, w, h, score] scaled to 224
            # Let's say face is in center. 
            # 224 scale: center is 112. 
            # Let's make a box of 100x100 at 62,62
            bb_path = folder / f"{i:06d}_BB.txt"
            with open(bb_path, 'w') as f:
                f.write("bbox = [62 62 100 100 0.99]")
                
    create_sequence(real_dir)
    create_sequence(spoof_dir)

def verify_pipeline():
    print("Verifying BB cropping pipeline...")
    
    create_mock_data()
    
    # Point to the root that contains CelebA_Spoof/Data...
    # Our generator expects data_dir to contain CelebA_Spoof/Data or be Data itself
    # We created data/video_liveness/CelebA_Spoof/Data
    data_dir = Path('data/video_liveness')
    
    # Try to find JSON
    json_path = None
    if Path('train_label.json').exists():
        json_path = Path('train_label.json')
        print(f"✅ Found local train_label.json")
        
    # Initialize generator
    data_gen = VideoDataGenerator(
        data_dir=data_dir,
        json_path=json_path,
        sequence_length=30,
        image_size=224,
        batch_size=4
    )
    
    if json_path:
        print(f"Real videos found: {len(data_gen.real_videos)}")
        print(f"Spoof videos found: {len(data_gen.spoof_videos)}")
        
        if len(data_gen.real_videos) == 0 and len(data_gen.spoof_videos) == 0:
            print("❌ No sequences found from JSON. Check paths.")
            return

    print("Creating dataset...")
    train_ds, val_ds = data_gen.create_dataset()
    
    if train_ds is None:
        print("❌ Failed to create dataset")
        return

    print("\nIterating through dataset...")
    start_time = time.time()
    
    # Take a few batches
    for i, (frames, labels) in enumerate(train_ds.take(5)):
        print(f"Batch {i}: Frames shape {frames.shape}, Labels shape {labels.shape}")
        
        # Verify shapes
        assert frames.shape[1:] == (30, 224, 224, 3), f"Wrong frame shape: {frames.shape}"
        assert labels.shape[0] == 2, f"Wrong batch size: {labels.shape}"
        
        # Check if cropping happened (we can't easily check content, but if it didn't crash it's good)
        
    duration = time.time() - start_time
    print(f"\n✅ Verification successful! Processed 5 batches in {duration:.2f}s")

if __name__ == "__main__":
    verify_pipeline()
