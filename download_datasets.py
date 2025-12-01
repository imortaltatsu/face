"""
CelebA-Spoof Dataset Downloader (using gdown)

1. Automatically installs gdown if missing.
2. Downloads the full CelebA-Spoof dataset folder from Google Drive.
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
    """Automated dataset downloader using gdown"""
    
    def __init__(self, output_dir='data/video_liveness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.celeba_dir = self.output_dir / 'celeba_spoof'
        # gdown downloads into the folder, so we point to the parent
        self.celeba_dir.mkdir(exist_ok=True)
        
    def download_celeba_spoof(self):
        """Download CelebA-Spoof using gdown folder download"""
        import gdown
        
        url = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing"
        print("\n" + "="*70)
        print("📥 Downloading CelebA-Spoof Dataset via gdown...")
        print("="*70)
        print(f"URL: {url}")
        print(f"Output: {self.celeba_dir}")
        
        try:
            # Download the folder
            # --folder flag equivalent
            gdown.download_folder(url, output=str(self.celeba_dir), quiet=False, use_cookies=False)
            print("\n✅ Download complete.")
            return True
        except Exception as e:
            print(f"\n❌ gdown failed: {e}")
            print("Possible reasons:")
            print("1. Google Drive rate limits (try again later)")
            print("2. Network issues")
            return False

    def unzip_files(self):
        """Unzip all zip files in the directory"""
        print(f"\n📦 Checking for zip files to extract in {self.celeba_dir}...")
        
        zip_files = sorted(list(self.celeba_dir.glob('**/*.zip')))
        
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
        
        if self.download_celeba_spoof():
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
