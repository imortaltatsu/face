"""
CelebA-Spoof Dataset Helper

1. Provides direct links for Google Drive and Baidu Drive.
2. Monitors directory for downloaded files.
3. Automatically unzips and organizes the dataset.
4. Handles multi-part zip files if present.
"""

import os
import sys
import zipfile
from pathlib import Path
import time
import shutil

class DatasetHelper:
    """Helper for managing CelebA-Spoof dataset"""
    
    def __init__(self, output_dir='data/video_liveness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.celeba_dir = self.output_dir / 'celeba_spoof'
        self.celeba_dir.mkdir(exist_ok=True)
        
    def show_instructions(self):
        """Print download instructions"""
        print("\n" + "="*70)
        print("📥 CelebA-Spoof Dataset Download Helper")
        print("="*70)
        print("\nAutomated download is restricted by file host protections.")
        print("Please download the dataset manually using one of the links below:")
        
        print("\nOPTION 1: Google Drive (Recommended)")
        print("🔗 Link: https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing")
        print("💡 Tip:  Use the 'Download all' button in Google Drive to get a single zip or split parts.")
        
        print("\nOPTION 2: Baidu Drive")
        print("🔗 Link:     https://pan.baidu.com/s/12qe13-jFJ9pE-_E3iSZtkw")
        print("🔑 Password: 61fd")
        
        print(f"\n📂 DESTINATION: {self.celeba_dir.absolute()}")
        print("\nINSTRUCTIONS:")
        print("1. Download the files (zip archives).")
        print(f"2. Move/Copy them to: {self.celeba_dir}")
        print("3. This script will automatically detect and unzip them.")
        print("="*70)
        
    def check_and_unzip(self):
        """Check for zip files and unzip them"""
        print(f"\n🔍 Checking for files in {self.celeba_dir}...")
        
        # Look for zip files
        zip_files = sorted(list(self.celeba_dir.glob('*.zip*')))
        
        if not zip_files:
            print("❌ No zip files found.")
            print("   Waiting for you to download files...")
            return False
            
        print(f"✅ Found {len(zip_files)} zip files.")
        
        # Filter out already extracted ones (heuristic)
        files_to_process = []
        for zf in zip_files:
            # Ignore partial parts like .z01, .z02 unless it's the main .zip
            if zf.suffix != '.zip' and '.zip' in zf.name:
                continue
                
            # Check if likely extracted
            possible_dir = zf.parent / zf.stem
            if possible_dir.exists() and possible_dir.is_dir():
                # print(f"   ℹ️  {zf.name} seems already extracted (folder exists).")
                pass
            else:
                files_to_process.append(zf)
        
        if not files_to_process:
            print("   All zip files appear to be extracted already.")
            return True

        for zip_path in files_to_process:
            print(f"\n📦 Processing {zip_path.name}...")
            try:
                print(f"   Unzipping to {zip_path.parent}...")
                
                # Handle multi-part zips if needed (basic support)
                # Usually standard zipfile handles it if you open the .zip file
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get list of files for progress bar
                    members = zip_ref.infolist()
                    for member in tqdm(members, desc="Extracting", unit="file"):
                        zip_ref.extract(member, zip_path.parent)
                        
                print(f"   ✅ Unzipped successfully")
                
                # Optional: Rename/Move logic could go here if structure is messy
                
            except zipfile.BadZipFile:
                print(f"   ❌ Error: {zip_path.name} is not a valid zip file (or download incomplete)")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        return True

    def run(self):
        """Main execution flow"""
        self.show_instructions()
        
        # Check immediately
        if self.check_and_unzip():
            print("\n✅ Dataset preparation complete!")
            return
            
        # Polling loop
        print("\nChecking again in 10 seconds... (Press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(10)
                if self.check_and_unzip():
                    print("\n✅ Dataset preparation complete!")
                    # Don't break immediately, keep watching for more files? 
                    # Or break if we think we are done. 
                    # Let's keep watching in case they download sequentially.
                    print("   Watching for more files...")
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    helper = DatasetHelper()
    helper.run()


if __name__ == '__main__':
    main()
