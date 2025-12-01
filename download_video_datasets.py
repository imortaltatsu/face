"""
Video Liveness Detection Dataset Downloader

Downloads and organizes video-based liveness detection datasets for training
micromovement detection models. Supports multiple datasets with verified links.
"""

import os
import sys
import requests
from pathlib import Path
import zipfile
import tarfile
from tqdm import tqdm
import json


class VideoLivenessDatasetDownloader:
    """Download and organize video liveness detection datasets"""
    
    def __init__(self, base_dir='data/video_liveness'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations with verified links
        self.datasets = {
            'replay_attack': {
                'name': 'Idiap Replay-Attack Database',
                'url': 'https://www.idiap.ch/en/scientific-research/data/replayattack',
                'type': 'manual',  # Requires registration
                'description': '1300 video clips of photo and video attacks on 50 clients',
                'size': '~3.5GB',
                'verified': True
            },
            'oulu_npu': {
                'name': 'OULU-NPU Face Liveness Detection',
                'url': 'https://sites.google.com/site/oulunpudatabase/',
                'type': 'manual',  # Requires EULA
                'description': '4950 real access and attack videos from 6 mobile devices',
                'size': '~5GB',
                'verified': True
            },
            'siw_mv2': {
                'name': 'SiW-Mv2 Face Anti-Spoofing',
                'url': 'https://github.com/CHELSEA234/SiW-Mv2',
                'type': 'manual',  # Requires DRA
                'description': 'Large-scale multi-view face anti-spoofing dataset',
                'size': '~10GB',
                'verified': True
            },
            'rose_youtu': {
                'name': 'ROSE-Youtu Face Liveness Detection',
                'url': 'http://rose1.ntu.edu.sg/Datasets/faceLivenessDetection.asp',
                'type': 'manual',
                'description': '4225 videos with 25 subjects, various attack types',
                'size': '~5.45GB',
                'verified': True
            },
            'celeba_spoof': {
                'name': 'CelebA-Spoof',
                'url': 'https://github.com/Davidzhangyuanhan/CelebA-Spoof',
                'type': 'manual',
                'description': '625,537 images of 10,177 subjects with spoof annotations',
                'size': '~50GB',
                'verified': True
            }
        }
    
    def list_datasets(self):
        """Display available datasets"""
        print("\n" + "="*70)
        print("📹 AVAILABLE VIDEO LIVENESS DETECTION DATASETS")
        print("="*70 + "\n")
        
        for idx, (key, dataset) in enumerate(self.datasets.items(), 1):
            status = "✅ Verified" if dataset['verified'] else "⚠️  Unverified"
            print(f"{idx}. {dataset['name']}")
            print(f"   Status: {status}")
            print(f"   Type: {dataset['type'].upper()}")
            print(f"   Size: {dataset['size']}")
            print(f"   Description: {dataset['description']}")
            print(f"   URL: {dataset['url']}")
            print()
    
    def download_instructions(self, dataset_key):
        """Provide download instructions for manual datasets"""
        dataset = self.datasets.get(dataset_key)
        if not dataset:
            print(f"❌ Dataset '{dataset_key}' not found")
            return
        
        print("\n" + "="*70)
        print(f"📥 DOWNLOAD INSTRUCTIONS: {dataset['name']}")
        print("="*70 + "\n")
        
        if dataset_key == 'replay_attack':
            print("1. Visit: https://www.idiap.ch/en/scientific-research/data/replayattack")
            print("2. Click 'Request Access' or 'Download'")
            print("3. Fill out the registration form")
            print("4. Wait for approval email with download link")
            print("5. Download the dataset")
            print(f"6. Extract to: {self.base_dir / 'replay_attack'}")
            
        elif dataset_key == 'oulu_npu':
            print("1. Visit: https://sites.google.com/site/oulunpudatabase/")
            print("2. Download the End User License Agreement (EULA)")
            print("3. Sign the EULA (requires permanent institutional position)")
            print("4. Send signed EULA from institutional email (no Gmail/Yahoo/Hotmail)")
            print("5. Wait for download link")
            print(f"6. Extract to: {self.base_dir / 'oulu_npu'}")
            
        elif dataset_key == 'siw_mv2':
            print("1. Visit: https://github.com/CHELSEA234/SiW-Mv2")
            print("2. Download the Dataset Release Agreement (DRA)")
            print("3. Sign the DRA")
            print("4. Email to: guoxia11@msu.edu")
            print("   Subject: SiW-Mv2 Dataset Request")
            print("   Include: Signed DRA, institution, purpose")
            print("5. Wait for download link")
            print(f"6. Extract to: {self.base_dir / 'siw_mv2'}")
            
        elif dataset_key == 'rose_youtu':
            print("1. Visit: http://rose1.ntu.edu.sg/Datasets/faceLivenessDetection.asp")
            print("2. Fill out the registration form")
            print("3. Accept the release agreement")
            print("4. Wait for approval and download link")
            print(f"5. Extract to: {self.base_dir / 'rose_youtu'}")
            
        elif dataset_key == 'celeba_spoof':
            print("1. Visit: https://github.com/Davidzhangyuanhan/CelebA-Spoof")
            print("2. Follow the download instructions in README")
            print("3. Use Google Drive links provided")
            print(f"4. Extract to: {self.base_dir / 'celeba_spoof'}")
        
        print("\n" + "="*70)
    
    def verify_dataset(self, dataset_key):
        """Check if dataset is downloaded"""
        dataset_path = self.base_dir / dataset_key
        if dataset_path.exists():
            file_count = len(list(dataset_path.rglob('*')))
            print(f"✅ Dataset '{dataset_key}' found: {file_count} files")
            return True
        else:
            print(f"❌ Dataset '{dataset_key}' not found at {dataset_path}")
            return False
    
    def create_synthetic_dataset(self, output_dir='data/synthetic_liveness'):
        """Create a synthetic dataset from webcam recordings"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("🎥 SYNTHETIC DATASET CREATION")
        print("="*70 + "\n")
        print("This will guide you through creating a synthetic liveness dataset")
        print("using your webcam for training.\n")
        
        print("Dataset structure:")
        print(f"  {output_dir}/")
        print("    ├── real/        # Real face videos (webcam)")
        print("    ├── print/       # Printed photo attacks")
        print("    ├── replay/      # Screen replay attacks")
        print("    └── mask/        # Mask attacks (optional)")
        
        print("\nInstructions:")
        print("1. Record 30-second videos of real faces (10+ people)")
        print("2. Record videos of printed photos being held up")
        print("3. Record videos of faces displayed on screens")
        print("4. Save videos in respective folders")
        print("5. Run preprocessing script to extract frames")
        
        # Create directory structure
        for category in ['real', 'print', 'replay', 'mask']:
            (output_path / category).mkdir(exist_ok=True)
        
        print(f"\n✅ Directory structure created at: {output_dir}")
        print("Ready for video collection!")


def main():
    """Main function with interactive menu"""
    downloader = VideoLivenessDatasetDownloader()
    
    while True:
        print("\n" + "="*70)
        print("🎬 VIDEO LIVENESS DETECTION DATASET MANAGER")
        print("="*70)
        print("\n1. List available datasets")
        print("2. Show download instructions")
        print("3. Verify downloaded dataset")
        print("4. Create synthetic dataset structure")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            downloader.list_datasets()
        
        elif choice == '2':
            print("\nAvailable datasets:")
            for idx, key in enumerate(downloader.datasets.keys(), 1):
                print(f"  {idx}. {key}")
            
            dataset_choice = input("\nEnter dataset name or number: ").strip()
            
            # Handle numeric input
            if dataset_choice.isdigit():
                idx = int(dataset_choice) - 1
                if 0 <= idx < len(downloader.datasets):
                    dataset_key = list(downloader.datasets.keys())[idx]
                else:
                    print("❌ Invalid number")
                    continue
            else:
                dataset_key = dataset_choice
            
            downloader.download_instructions(dataset_key)
        
        elif choice == '3':
            dataset_key = input("Enter dataset name: ").strip()
            downloader.verify_dataset(dataset_key)
        
        elif choice == '4':
            downloader.create_synthetic_dataset()
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == '__main__':
    main()
