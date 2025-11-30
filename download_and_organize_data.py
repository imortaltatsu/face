"""
Download and Organize Face Anti-Spoofing Datasets

This comprehensive script handles:
1. Downloading datasets from Google Drive (NUAA, CASIA-SURF, etc.)
2. Organizing datasets into train/val splits
3. Merging multiple datasets
4. Collecting your own data via webcam

All datasets are automatically resized to 96x96 and merged into unified train/val sets.
"""

import os
from pathlib import Path
import zipfile
from tqdm import tqdm
import cv2
import numpy as np
from datetime import datetime

# Hardcoded dataset links (image-based only)
DATASETS = {
    'nuaa': {
        'name': 'NUAA Photograph Imposter',
        'gdrive_id': '1-aSGKdAIK0YoKxQvnNx1KJvTm4zbwZLz',
        'description': 'Real faces vs printed photos, ~12K images',
        'size': '~400MB',
        'type': 'gdrive'
    },
    'celeba_spoof': {
        'name': 'CelebA-Spoof',
        'url': 'https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z',
        'description': 'High-quality spoofing dataset, 625K images (multiple zips)',
        'size': '~10GB',
        'type': 'manual',
        'note': 'Google Drive folder - download all zips manually'
    },
    'msu_mfsd': {
        'name': 'MSU-MFSD',
        'url': 'https://www.cse.msu.edu/rgroups/biometrics/Publications/Databases/MSUMobileFaceSpoofing/MSU_MFSD.zip',
        'description': 'Mobile face spoofing, print & replay attacks',
        'size': '~2GB',
        'type': 'direct_url'
    }
}


class DatasetManager:
    """Comprehensive dataset download, organization, and merging"""
    
    def __init__(self, data_dir='data/liveness'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        
        # Create directories
        for d in [self.raw_dir, self.train_dir, self.val_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "=" * 60)
        print("Face Anti-Spoofing Dataset Manager")
        print("=" * 60)
        
        # Show current dataset status
        train_real = len(list((self.train_dir / 'real').glob('*.jpg')))
        train_fake = len(list((self.train_dir / 'fake').glob('*.jpg')))
        val_real = len(list((self.val_dir / 'real').glob('*.jpg')))
        val_fake = len(list((self.val_dir / 'fake').glob('*.jpg')))
        
        if train_real > 0 or train_fake > 0:
            print(f"\n📊 Current Dataset:")
            print(f"   Training: {train_real:,} real + {train_fake:,} fake = {train_real + train_fake:,} total")
            print(f"   Validation: {val_real:,} real + {val_fake:,} fake = {val_real + val_fake:,} total")
            print(f"   Grand Total: {train_real + train_fake + val_real + val_fake:,} images")
        else:
            print("\n📊 No dataset found - let's get started!")
        
        print("\n" + "=" * 60)
        print("Available Options:")
        print("=" * 60)
        
        print("\n📥 Download & Organize:")
        print("  1. Download NUAA dataset (recommended - 12K images)")
        print("  2. Show CelebA-Spoof info (manual download - 625K images)")
        print("  3. Download MSU-MFSD dataset (2GB)")
        print("  4. Download ALL available datasets")
        print("  5. Download from custom Google Drive link")
        
        print("\n📂 Organize Existing:")
        print("  6. Organize manually downloaded dataset")
        
        print("\n🎥 Collect Your Own:")
        print("  7. Collect real faces via webcam")
        print("  8. Collect fake faces (photos) via webcam")
        
        print("\n📊 Info:")
        print("  9. Show dataset summary")
        print("  10. Exit")
        
        print("\n" + "=" * 60)
    
    def download_from_gdrive(self, file_id_or_url, output_name='dataset'):
        """Download dataset from Google Drive using curl (more reliable than gdown)"""
        import subprocess
        
        # Extract file ID from URL if needed
        if 'drive.google.com' in file_id_or_url:
            if '/d/' in file_id_or_url:
                file_id = file_id_or_url.split('/d/')[1].split('/')[0]
            elif 'id=' in file_id_or_url:
                file_id = file_id_or_url.split('id=')[1].split('&')[0]
            else:
                print("❌ Could not extract file ID from URL")
                return None
        else:
            file_id = file_id_or_url
        
        print(f"\n📥 Downloading from Google Drive (ID: {file_id})...")
        
        output_file = self.raw_dir / f'{output_name}.zip'
        
        try:
            # Use curl with Google Drive's direct download URL (handles large files)
            url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
            
            print(f"   Using curl for reliable download...")
            
            # Run curl command
            result = subprocess.run(
                ['curl', '-L', url, '-o', str(output_file)],
                capture_output=True,
                text=True
            )
            
            # If SSL error, retry with --insecure
            if result.returncode != 0 and 'SSL certificate' in result.stderr:
                print(f"   ⚠️  SSL error detected, retrying with --insecure flag...")
                result = subprocess.run(
                    ['curl', '-k', '-L', url, '-o', str(output_file)],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                raise Exception(f"curl failed: {result.stderr}")
            
            # Check if file was downloaded
            if not output_file.exists() or output_file.stat().st_size < 1000:
                raise Exception("Downloaded file is too small or doesn't exist")
            
            print(f"✅ Downloaded to: {output_file}")
            print(f"   Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Extract
            print(f"\n📦 Extracting...")
            extract_dir = self.raw_dir / output_name
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                for member in tqdm(zip_ref.filelist, desc='Extracting'):
                    zip_ref.extract(member, extract_dir)
            
            print(f"✅ Extracted to: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Download failed: {e}")
            print(f"\n🔗 Attempted URL: https://drive.google.com/uc?id={file_id}")
            print(f"📋 File ID: {file_id}")
            
            # Check for SSL errors
            if 'SSL' in error_msg or 'CERTIFICATE' in error_msg:
                print("\n⚠️  SSL Certificate Error Detected!")
                print("\n💡 SSL Troubleshooting:")
                print("   1. Install/update certificates:")
                print("      macOS: /Applications/Python*/Install\\ Certificates.command")
                print("      or: pip install --upgrade certifi")
                print("   2. Try with SSL verification disabled (not recommended):")
                print("      Set environment variable: PYTHONHTTPSVERIFY=0")
                print("   3. Download manually from:")
                print(f"      https://drive.google.com/file/d/{file_id}/view")
            else:
                print("\n💡 Troubleshooting:")
                print("   1. Check if the file is publicly accessible")
                print("   2. Try downloading manually from:")
                print(f"      https://drive.google.com/file/d/{file_id}/view")
                print("   3. If manual download works, use option 6 to organize")
            
            return None
    
    def download_from_url(self, url, output_name='dataset'):
        """Download dataset from direct URL"""
        import requests
        
        print(f"\n📥 Downloading from {url}...")
        
        output_file = self.raw_dir / f'{output_name}.zip'
        
        try:
            # Try with SSL verification first
            try:
                response = requests.get(url, stream=True, timeout=30, verify=True)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                print("   ⚠️  SSL error detected, retrying without verification...")
                response = requests.get(url, stream=True, timeout=30, verify=False)
                response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, download without progress
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✅ Downloaded to: {output_file}")
            
            # Extract
            print(f"\n📦 Extracting...")
            extract_dir = self.raw_dir / output_name
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                for member in tqdm(zip_ref.filelist, desc='Extracting'):
                    zip_ref.extract(member, extract_dir)
            
            print(f"✅ Extracted to: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Download failed: {e}")
            print(f"\n🔗 Attempted URL: {url}")
            
            # Check for SSL errors
            if 'SSL' in error_msg or 'CERTIFICATE' in error_msg:
                print("\n⚠️  SSL Certificate Error Detected!")
                print("\n💡 SSL Troubleshooting:")
                print("   1. Install/update certificates:")
                print("      macOS: /Applications/Python*/Install\\ Certificates.command")
                print("      or: pip install --upgrade certifi")
                print("   2. Update requests library:")
                print("      uv add --upgrade requests")
                print("   3. Try downloading with curl:")
                print(f"      curl -L '{url}' -o dataset.zip")
            else:
                print("\n💡 Troubleshooting:")
                print("   1. Check your internet connection")
                print("   2. Verify the URL is still valid:")
                print(f"      {url}")
                print("   3. Try downloading manually and use option 6 to organize")
                print("   4. The dataset might have moved - check for updated links")
            
            return None

    
    def organize_dataset(self, dataset_dir, dataset_name, train_split=0.8):
        """
        Organize any dataset with real/fake structure
        
        Automatically finds:
        - real/live/genuine/client directories
        - fake/spoof/attack/imposter directories
        """
        print(f"\n📂 Organizing {dataset_name}...")
        
        # Find real and fake directories
        real_patterns = ['*real*', '*live*', '*genuine*', '*client*', '*ClientRaw*']
        fake_patterns = ['*fake*', '*spoof*', '*attack*', '*imposter*', '*ImposterRaw*']
        
        real_dirs = []
        fake_dirs = []
        
        for pattern in real_patterns:
            real_dirs.extend([d for d in dataset_dir.rglob(pattern) if d.is_dir()])
        
        for pattern in fake_patterns:
            fake_dirs.extend([d for d in dataset_dir.rglob(pattern) if d.is_dir()])
        
        if not real_dirs or not fake_dirs:
            print(f"⚠️  Could not auto-detect real/fake directories")
            print("   Available directories:")
            for d in dataset_dir.rglob('*'):
                if d.is_dir() and not d.name.startswith('.'):
                    print(f"     - {d.relative_to(dataset_dir)}")
            
            print("\n   Please organize manually:")
            print(f"     Real faces → {self.train_dir / 'real'}")
            print(f"     Fake faces → {self.train_dir / 'fake'}")
            return False
        
        # Use first found directories
        real_dir = real_dirs[0]
        fake_dir = fake_dirs[0]
        
        print(f"   Real: {real_dir.name}")
        print(f"   Fake: {fake_dir.name}")
        
        # Collect images
        real_images = []
        fake_images = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
            real_images.extend(list(real_dir.rglob(ext)))
            fake_images.extend(list(fake_dir.rglob(ext)))
        
        print(f"   Found {len(real_images)} real, {len(fake_images)} fake images")
        
        if not real_images or not fake_images:
            print("❌ No images found!")
            return False
        
        # Merge into existing dataset
        self._merge_images(real_images, fake_images, dataset_name, train_split)
        
        return True
    
    def _merge_images(self, real_images, fake_images, dataset_name, train_split=0.8):
        """Merge images into existing train/val splits"""
        
        # Shuffle
        np.random.shuffle(real_images)
        np.random.shuffle(fake_images)
        
        # Split
        real_split = int(len(real_images) * train_split)
        fake_split = int(len(fake_images) * train_split)
        
        # Get existing counts for unique naming
        existing_train_real = len(list((self.train_dir / 'real').glob('*.jpg')))
        existing_train_fake = len(list((self.train_dir / 'fake').glob('*.jpg')))
        existing_val_real = len(list((self.val_dir / 'real').glob('*.jpg')))
        existing_val_fake = len(list((self.val_dir / 'fake').glob('*.jpg')))
        
        print(f"\n📋 Merging {dataset_name} into training set...")
        
        # Create target directories
        (self.train_dir / 'real').mkdir(exist_ok=True)
        (self.train_dir / 'fake').mkdir(exist_ok=True)
        (self.val_dir / 'real').mkdir(exist_ok=True)
        (self.val_dir / 'fake').mkdir(exist_ok=True)
        
        # Training set
        self._copy_images(
            real_images[:real_split],
            self.train_dir / 'real',
            f'{dataset_name}_train_real',
            start_idx=existing_train_real
        )
        self._copy_images(
            fake_images[:fake_split],
            self.train_dir / 'fake',
            f'{dataset_name}_train_fake',
            start_idx=existing_train_fake
        )
        
        # Validation set
        self._copy_images(
            real_images[real_split:],
            self.val_dir / 'real',
            f'{dataset_name}_val_real',
            start_idx=existing_val_real
        )
        self._copy_images(
            fake_images[fake_split:],
            self.val_dir / 'fake',
            f'{dataset_name}_val_fake',
            start_idx=existing_val_fake
        )
    
    def _copy_images(self, images, target_dir, prefix, start_idx=0):
        """Copy and resize images to 544x544"""
        for i, img_path in enumerate(tqdm(images, desc=f"Adding {prefix}")):
            target_path = target_dir / f'{prefix}_{start_idx + i:05d}.jpg'
            
            img = cv2.imread(str(img_path))
            if img is not None:
                img_resized = cv2.resize(img, (544, 544))
                cv2.imwrite(str(target_path), img_resized)
    
    def collect_webcam_data(self, data_type='real', num_samples=100):
        """Collect data via webcam"""
        if data_type == 'real':
            output_dir = self.train_dir / 'real'
            title = "Collecting REAL faces"
            instructions = [
                "Look at camera naturally",
                "Move head slightly (different angles)",
                "Try different expressions",
                "Vary lighting if possible"
            ]
            color = (0, 255, 0)  # Green
        else:
            output_dir = self.train_dir / 'fake'
            title = "Collecting FAKE faces (photos)"
            instructions = [
                "Hold a PHOTO of a face in front of camera",
                "OR display photo on phone/tablet screen",
                "Try different angles and distances",
                "Use different photos of different people"
            ]
            color = (0, 0, 255)  # Red
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print("\nInstructions:")
        for inst in instructions:
            print(f"  - {inst}")
        print(f"\nPress SPACE to capture ({num_samples} samples needed)")
        print("Press 'q' to quit\n")
        
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display
            cv2.putText(frame, f"{data_type.upper()} Faces: {count}/{num_samples}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "Press SPACE to capture", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(f'Collect {data_type.upper()} Faces', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = output_dir / f'{data_type}_{timestamp}.jpg'
                
                # Resize to 544x544
                face_resized = cv2.resize(frame, (544, 544))
                cv2.imwrite(str(filename), face_resized)
                
                count += 1
                print(f"✓ Captured {count}/{num_samples}")
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {count} {data_type} face samples")
    
    def print_summary(self):
        """Print dataset summary"""
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        
        train_real = len(list((self.train_dir / 'real').glob('*.jpg')))
        train_fake = len(list((self.train_dir / 'fake').glob('*.jpg')))
        val_real = len(list((self.val_dir / 'real').glob('*.jpg')))
        val_fake = len(list((self.val_dir / 'fake').glob('*.jpg')))
        
        print(f"\nTraining Set:")
        print(f"  Real: {train_real:,} images")
        print(f"  Fake: {train_fake:,} images")
        print(f"  Total: {train_real + train_fake:,} images")
        
        print(f"\nValidation Set:")
        print(f"  Real: {val_real:,} images")
        print(f"  Fake: {val_fake:,} images")
        print(f"  Total: {val_real + val_fake:,} images")
        
        print(f"\nGrand Total: {train_real + train_fake + val_real + val_fake:,} images")
        
        print("\n" + "=" * 60)
        
        if train_real > 0 and train_fake > 0:
            print("✅ Ready to train!")
            print("=" * 60)
            print("\nRun: python train.py")
        else:
            print("⚠️  Need more data to train")
            print("=" * 60)


def main():
    """Main dataset management pipeline"""
    manager = DatasetManager()
    
    while True:
        manager.show_menu()
        choice = input("\nEnter choice (1-10): ").strip()
        
        if choice == '1':
            # Download NUAA
            extract_dir = manager.download_from_gdrive(
                DATASETS['nuaa']['gdrive_id'],
                'nuaa'
            )
            if extract_dir:
                raw_subdir = extract_dir / 'raw'
                if raw_subdir.exists():
                    extract_dir = raw_subdir
                manager.organize_dataset(extract_dir, 'NUAA')
                manager.print_summary()
        
        elif choice == '2':
            # Show CelebA-Spoof info
            info = DATASETS['celeba_spoof']
            print(f"\n📦 {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   ⚠️  {info['note']}")
            print(f"\n   Download from: {info['url']}")
            print("   After downloading, use option 6 to organize")
        
        elif choice == '3':
            # Download MSU-MFSD
            extract_dir = manager.download_from_url(
                DATASETS['msu_mfsd']['url'],
                'msu_mfsd'
            )
            if extract_dir:
                manager.organize_dataset(extract_dir, 'MSU-MFSD')
                manager.print_summary()
        
        elif choice == '4':
            # Download all
            for key, info in DATASETS.items():
                if info.get('type') == 'manual':
                    print(f"\n⏭️  Skipping {info['name']} (requires manual download)")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Downloading {info['name']}")
                print(f"{'='*60}")
                
                if info.get('type') == 'gdrive':
                    extract_dir = manager.download_from_gdrive(info['gdrive_id'], key)
                else:
                    extract_dir = manager.download_from_url(info['url'], key)
                
                if extract_dir:
                    manager.organize_dataset(extract_dir, info['name'])
            manager.print_summary()
        
        elif choice == '5':
            # Custom Google Drive link
            print("\n📥 Custom Google Drive Download")
            gdrive_input = input("Paste Google Drive URL or file ID: ").strip()
            dataset_name = input("Dataset name: ").strip() or "custom"
            
            if gdrive_input:
                extract_dir = manager.download_from_gdrive(gdrive_input, dataset_name)
                if extract_dir:
                    manager.organize_dataset(extract_dir, dataset_name)
                    manager.print_summary()
        
        elif choice == '6':
            # Organize manual download
            print("\n📁 Organize Manually Downloaded Dataset")
            dataset_path = input("Enter path to dataset: ").strip()
            dataset_name = input("Dataset name: ").strip() or "manual"
            
            if Path(dataset_path).exists():
                manager.organize_dataset(Path(dataset_path), dataset_name)
                manager.print_summary()
            else:
                print(f"❌ Path not found: {dataset_path}")
        
        elif choice == '7':
            # Collect real faces
            num_samples = int(input("\nHow many real face samples? (default: 100): ") or "100")
            manager.collect_webcam_data('real', num_samples)
            manager.print_summary()
        
        elif choice == '8':
            # Collect fake faces
            num_samples = int(input("\nHow many fake face samples? (default: 100): ") or "100")
            manager.collect_webcam_data('fake', num_samples)
            manager.print_summary()
        
        elif choice == '9':
            # Show summary
            manager.print_summary()
        
        elif choice == '10':
            print("\n👋 Exiting...")
            break
        
        else:
            print("\n❌ Invalid choice")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
