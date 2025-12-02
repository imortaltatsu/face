import os
from pathlib import Path
import sys

def check_dataset():
    print("🔍 Checking CelebA-Spoof Dataset...")
    
    # Check paths
    base_dir = Path('data/video_liveness')
    dataset_dir = base_dir / 'celeba_spoof' / 'CelebA_Spoof'
    data_dir = dataset_dir / 'Data'
    
    print(f"Base Dir: {base_dir.absolute()}")
    print(f"Dataset Dir: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        # Try finding where it might be
        print("Listing base_dir contents:")
        if base_dir.exists():
            for p in base_dir.iterdir():
                print(f"  - {p.name}")
        return

    # Check zip files
    zip_files = sorted(list(base_dir.glob('**/*.zip.*')))
    print(f"\n📦 Found {len(zip_files)} zip parts.")
    
    expected_parts = 74
    missing_parts = []
    for i in range(1, expected_parts + 1):
        part_name = f"CelebA_Spoof.zip.{i:03d}"
        if not any(z.name == part_name for z in zip_files):
            missing_parts.append(part_name)
            
    if len(zip_files) == expected_parts:
        print("✅ All 74 zip parts found.")
    else:
        print(f"❌ Missing {len(missing_parts)} parts.")
        if len(missing_parts) < 10:
            print(f"   Missing: {missing_parts}")
        else:
            print(f"   Missing (first 10): {missing_parts[:10]}...")
            
    if len(zip_files) > 0:
        print(f"   First found: {zip_files[0].name}")
        print(f"   Last found:  {zip_files[-1].name}")
    
    # Check extracted data
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
        
    print(f"\n📂 Checking Data directory: {data_dir}")
    
    # Count subjects
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    train_subjects = list(train_dir.iterdir()) if train_dir.exists() else []
    test_subjects = list(test_dir.iterdir()) if test_dir.exists() else []
    
    print(f"   Train subjects: {len(train_subjects)}")
    print(f"   Test subjects:  {len(test_subjects)}")
    
    # Count sequences (live/spoof folders with PNGs)
    print("\n🔄 Counting sequences (this might take a moment)...")
    
    real_count = 0
    spoof_count = 0
    total_pngs = 0
    
    # Sample check - don't walk everything if it's huge, or do?
    # Let's walk but print progress
    
    for root, dirs, files in os.walk(data_dir):
        if 'live' in os.path.basename(root):
            pngs = [f for f in files if f.endswith('.png')]
            if pngs:
                real_count += 1
                total_pngs += len(pngs)
        elif 'spoof' in os.path.basename(root):
            pngs = [f for f in files if f.endswith('.png')]
            if pngs:
                spoof_count += 1
                total_pngs += len(pngs)
                
        if (real_count + spoof_count) % 1000 == 0 and (real_count + spoof_count) > 0:
            print(f"   Found {real_count + spoof_count} sequences...", end='\r')
            
    print(f"\n✅ Scan complete.")
    print(f"   Real sequences:  {real_count}")
    print(f"   Spoof sequences: {spoof_count}")
    print(f"   Total sequences: {real_count + spoof_count}")
    print(f"   Total PNGs:      {total_pngs}")
    
    if real_count + spoof_count < 100:
        print("\n⚠️  WARNING: Very few sequences found!")
        print("   Possible reasons:")
        print("   1. Extraction failed or was interrupted.")
        print("   2. Dataset path is incorrect.")
        print("   3. Zip files were not fully downloaded.")

if __name__ == "__main__":
    check_dataset()
