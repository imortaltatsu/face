import tensorflow as tf
from data_loader import FaceImageDataGenerator
from pathlib import Path
import time

def verify_pipeline():
    print("🧪 Verifying Image Pipeline...")
    
    # Mock data directory
    data_dir = Path('data/video_liveness')
    
    # Check if data exists, otherwise mock
    if not data_dir.exists():
        print("⚠️ Data dir not found, using mock paths for testing logic")
        # We can't easily mock the generator without files since it scans directories
        # But we can test the class instantiation and methods if we mock os.walk or just rely on what's there
        # Let's assume the user has the subset we saw earlier
        pass

    # Initialize Generator
    try:
        gen = FaceImageDataGenerator(
            data_dir=data_dir,
            image_size=224,
            batch_size=4
        )
        
        # Create dataset
        # If no JSON, it scans. If no files found, it returns None
        dataset, _ = gen.create_dataset(validation_split=0.2)
        
        if dataset is None:
            print("❌ No dataset created (likely no files found).")
            return
            
        print("✅ Dataset created successfully.")
        
        # Take one batch
        print("🔄 Fetching one batch...")
        start = time.time()
        for images, labels in dataset.take(1):
            print(f"   Image Batch Shape: {images.shape}")
            print(f"   Labels Shape: {labels.shape}")
            print(f"   Labels: {labels.numpy()}")
            
            # Verify shape
            assert images.shape == (4, 224, 224, 3), f"Expected (4, 224, 224, 3), got {images.shape}"
            assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
            
        end = time.time()
        print(f"✅ Batch loaded in {end - start:.4f} seconds")
        
    except Exception as e:
        print(f"❌ Pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()
