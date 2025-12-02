"""
CelebA-Spoof Dataset Downloader (ID-based)

1. Automatically installs gdown if missing.
2. Downloads CelebA-Spoof dataset files using specific Google Drive File IDs.
   (Bypasses folder download limits)
3. Automatically unzips downloaded archives.
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import shutil

def install_gdown():
    """Install gdown if not present"""
    try:
        import gdown
        print("✅ gdown is already installed.")
    except ImportError:
        print("📦 Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✅ gdown installed successfully.")
        except Exception as e:
            print(f"❌ Failed to install gdown: {e}")
            sys.exit(1)

class DatasetDownloader:
    """Automated dataset downloader using gdown with specific File IDs"""
    
    def __init__(self, output_dir='data/video_liveness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.celeba_dir = self.output_dir / 'celeba_spoof'
        self.celeba_dir.mkdir(exist_ok=True)
        
        # File IDs found from community resources (GitHub Gist)
        # These correspond to the split zip files of the dataset
        self.file_ids = [
            '1gd0M4qpxzqrnOkCl7hv7vnCWV4JJTFwM',
            '1OsOiow42GS4wbE7o1csPA2HtKLSgXT5_',
            '1_07Q7VA4-4R5fvHpkyoorhD5igeathko',
            '1Oy-5J10hsrZ2gyHH9U2oet0F92oaEqSq',
            '1NxlMSgvJSlDJMRFIfq_cT_RRybYS8eNW',
            '1mpMTb5ODq-9NV4IEVFwEDhUbYSNfVQtC',
            '14d1TYmcVXg1TlmuywM0HgOKcPqa71Rmq',
            '1st5Yh8yRQGAmI02iueWPm2rrJN1TpOlC',
            '1K-6UwtmUvrW7sZqaUvGYIkqpPC1G3K1V',
            '1wNgEJSBlZuQKePCTbwLM_9jBuIg3Hj1f',
            '18qd2y1fBPiqQqgSM-vrQq2rMbXeSHWLy',
            '1HgwT6-NLplcBkowdSHQz5VZEGEsFFDRd',
            '1-UYx3LoCloTNchmSq5otd4GxkWVTEGs_',
            '1MiHlWChmvOy53eTyiwWvd6hRV8zSeEYJ',
            '1WqikxT7XCMZLaPaYfOo0gYZOxeqAiFcP',
            '1iswrU0QuYSbzwKk0g0TWDD52hbtOps21',
            '1JPAB1jjLVdn8BBJk4hydDZq-IV5czrGq'
        ]
        
    def download_files(self):
        """Download files individually using gdown"""
        import gdown
        
        print("\n" + "="*70)
        print("📥 Downloading CelebA-Spoof Dataset Parts...")
        print("="*70)
        print(f"Target Directory: {self.celeba_dir}")
        print(f"Total Files: {len(self.file_ids)}")
        
        success_count = 0
        
        for i, file_id in enumerate(self.file_ids):
            print(f"\n[{i+1}/{len(self.file_ids)}] Downloading file ID: {file_id}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Output filename will be determined by gdown automatically
            # or we can let it download to the dir
            try:
                # We change cwd temporarily to download into the target folder
                # or use output parameter if we knew the name. 
                # gdown handles names well.
                
                output_path = gdown.download(url, output=None, quiet=False, verify=False)
                
                if output_path:
                    # Move to target dir if not already there
                    # gdown downloads to CWD by default if output is None
                    filename = os.path.basename(output_path)
                    target_path = self.celeba_dir / filename
                    
                    if os.path.abspath(output_path) != str(target_path.absolute()):
                        shutil.move(output_path, target_path)
                        print(f"   Moved to {target_path}")
                    
                    success_count += 1
                else:
                    print("   ❌ Download failed (no output path)")
                    
            except Exception as e:
                print(f"   ❌ Error downloading {file_id}: {e}")
                
        print(f"\n✅ Downloaded {success_count}/{len(self.file_ids)} files.")
        return success_count > 0

    def unzip_files(self):
        """Unzip all zip files in the directory"""
        print(f"\n📦 Checking for zip files to extract in {self.celeba_dir}...")
        
        zip_files = sorted(list(self.celeba_dir.glob('*.zip')))
        
        if not zip_files:
            print("   No zip files found.")
            return

        print(f"   Found {len(zip_files)} zip files.")
        
        for zip_path in zip_files:
            # Check if likely already extracted
            extract_dir = zip_path.parent / zip_path.stem
            if extract_dir.exists() and extract_dir.is_dir():
                # print(f"   ℹ️  {zip_path.name} seems extracted. Skipping.")
                continue
                
            print(f"   Unzipping {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_path.parent)
                print(f"   ✅ Extracted")
            except zipfile.BadZipFile:
                print(f"   ❌ Error: Invalid zip file")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    def run(self):
        """Main execution flow"""
        install_gdown()
        
        if self.download_files():
            self.unzip_files()
            print("\n" + "="*70)
            print("✅ DATASET PREPARATION COMPLETE")
            print("="*70)
        else:
            print("\n❌ Dataset preparation failed.")


def main():
    downloader = DatasetDownloader()
    downloader.run()


if __name__ == '__main__':
    main()
