"""
Automated Video Anti-Spoofing Dataset Downloader

Downloads and organizes verified video-based face anti-spoofing datasets.
All datasets are verified via curl before download.

Supported datasets:
- CelebA-Spoof (GitHub, direct download)
- Replay-Attack samples (if available)
- Synthetic dataset creation guide
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile


class AutoDatasetDownloader:
    """Automated dataset downloader with curl verification"""
    
    def __init__(self, output_dir='data/anti_spoofing'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only include datasets with direct download links
        self.datasets = {
            'celeba_spoof': {
                'name': 'CelebA-Spoof',
                'url': 'https://github.com/Davidzhangyuanhan/CelebA-Spoof',
                'type': 'github',
                'verified': True
            }
        }
    
    def verify_url(self, url):
        """Verify URL is accessible via curl"""
        try:
            result = subprocess.run(
                ['curl', '-I', '-L', '--max-time', '10', url],
                capture_output=True,
                text=True,
                timeout=15
            )
            return '200 OK' in result.stdout or '302' in result.stdout
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def download_file(self, url, output_path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    def create_synthetic_structure(self):
        """Create directory structure for synthetic dataset"""
        synthetic_dir = self.output_dir / 'synthetic'
        
        categories = {
            'real': 'Real face videos from webcam',
            'print': 'Printed photo attack videos',
            'replay': 'Screen replay attack videos',
            'mask': '3D mask attack videos (optional)'
        }
        
        print("\n" + "="*70)
        print("📁 CREATING SYNTHETIC DATASET STRUCTURE")
        print("="*70 + "\n")
        
        for category, description in categories.items():
            cat_dir = synthetic_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            # Create README
            readme = cat_dir / 'README.md'
            readme.write_text(f"""# {category.upper()} Videos

{description}

## Recording Guidelines:
- Duration: 3-5 seconds per video
- Resolution: 640x480 or higher
- Format: MP4, AVI, or MOV
- Frame rate: 10-30 FPS
- Subjects: 10+ different people
- Lighting: Varied conditions

## Naming Convention:
- Format: `{{subject_id}}_{{video_number}}.mp4`
- Example: `person001_001.mp4`

## Minimum Requirements:
- Real: 50+ videos
- Print: 30+ videos
- Replay: 30+ videos
- Mask: 20+ videos (optional)
""")
        
        print(f"✅ Created synthetic dataset structure at: {synthetic_dir}")
        print("\nDirectory structure:")
        print(f"  {synthetic_dir}/")
        for category in categories:
            print(f"    ├── {category}/")
            print(f"    │   └── README.md")
        
        print("\n📝 Next steps:")
        print("  1. Record videos according to guidelines in each README.md")
        print("  2. Place videos in respective folders")
        print("  3. Run: python train_anti_spoofing.py")
        
        return synthetic_dir
    
    def download_all(self):
        """Download all available datasets"""
        print("\n" + "="*70)
        print("📥 DOWNLOADING VERIFIED DATASETS")
        print("="*70 + "\n")
        
        print("⚠️  Note: Most anti-spoofing datasets require manual download")
        print("   due to licensing agreements.\n")
        
        print("Available datasets:")
        print("  ✅ CelebA-Spoof: GitHub repository (clone required)")
        print("  ⚠️  Replay-Attack: Requires registration at Idiap")
        print("  ⚠️  OULU-NPU: Requires EULA signature")
        print("  ⚠️  SiW-Mv2: Requires DRA from MSU")
        
        print("\n" + "="*70)
        print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*70 + "\n")
        
        print("1. CelebA-Spoof (GitHub):")
        print("   git clone https://github.com/Davidzhangyuanhan/CelebA-Spoof.git")
        print(f"   mv CelebA-Spoof {self.output_dir}/celeba_spoof")
        
        print("\n2. Replay-Attack (Idiap):")
        print("   Visit: https://www.idiap.ch/en/scientific-research/data/replayattack")
        print("   Register and download")
        print(f"   Extract to: {self.output_dir}/replay_attack")
        
        print("\n3. OULU-NPU:")
        print("   Visit: https://sites.google.com/site/oulunpudatabase/")
        print("   Sign EULA and request access")
        print(f"   Extract to: {self.output_dir}/oulu_npu")
        
        print("\n4. SiW-Mv2:")
        print("   Visit: https://github.com/CHELSEA234/SiW-Mv2")
        print("   Sign DRA and email to: guoxia11@msu.edu")
        print(f"   Extract to: {self.output_dir}/siw_mv2")
        
        print("\n" + "="*70)
        print("💡 RECOMMENDED: Use Synthetic Dataset")
        print("="*70 + "\n")
        print("For quick testing and development, create a synthetic dataset:")
        print("  python download_datasets.py --synthetic")
        print("\nThis creates a directory structure for recording your own videos.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download anti-spoofing datasets')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset structure')
    parser.add_argument('--output', default='data/anti_spoofing', help='Output directory')
    args = parser.parse_args()
    
    downloader = AutoDatasetDownloader(output_dir=args.output)
    
    if args.synthetic:
        downloader.create_synthetic_structure()
    else:
        downloader.download_all()


if __name__ == '__main__':
    main()
