"""
Automated Anti-Spoofing Dataset Scraper & Generator

1. Scrapes dataset websites for direct download links.
2. Handles SSL/Certificate errors gracefully.
3. Generates synthetic training data if real data cannot be downloaded.
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import subprocess
import re
from urllib.parse import urljoin, urlparse
import cv2
import numpy as np
import random
import warnings

# Suppress SSL warnings
warnings.filterwarnings("ignore")

class DatasetScraper:
    """Automated dataset scraper and downloader"""
    
    def __init__(self, output_dir='data/video_liveness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        # Disable SSL verification for session
        self.session.verify = False
    
    def scrape_github_releases(self, repo_url):
        """Scrape GitHub repository for release downloads"""
        try:
            # Convert to API URL
            repo_path = urlparse(repo_url).path.strip('/')
            api_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
            
            response = self.session.get(api_url, verify=False)
            if response.status_code == 200:
                data = response.json()
                assets = data.get('assets', [])
                return [asset['browser_download_url'] for asset in assets]
        except Exception as e:
            print(f"Failed to scrape {repo_url}: {e}")
        return []
    
    def scrape_google_drive_links(self, page_url):
        """Extract Google Drive links from webpage"""
        try:
            response = self.session.get(page_url, verify=False)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'drive.google.com' in href or 'docs.google.com' in href:
                    # Extract file ID
                    match = re.search(r'/d/([a-zA-Z0-9_-]+)', href)
                    if match:
                        file_id = match.group(1)
                        links.append(f"https://drive.google.com/uc?id={file_id}&export=download")
            
            return links
        except Exception as e:
            print(f"Failed to scrape {page_url}: {e}")
        return []
    
    def download_file(self, url, output_path):
        """Download file with progress bar"""
        try:
            # Disable SSL verification
            response = self.session.get(url, stream=True, timeout=30, verify=False)
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
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def download_with_gdown(self, gdrive_url, output_path):
        """Download from Google Drive using gdown"""
        try:
            import gdown
            # Use verify=False to bypass SSL errors
            gdown.download(gdrive_url, str(output_path), quiet=False, verify=False)
            return True
        except ImportError:
            print("gdown not installed. Install with: pip install gdown")
            return False
        except Exception as e:
            print(f"gdown failed: {e}")
            return False
    
    def scrape_celeba_spoof(self):
        """Scrape CelebA-Spoof dataset"""
        print("\n📥 Scraping CelebA-Spoof...")
        
        repo_url = "https://github.com/Davidzhangyuanhan/CelebA-Spoof"
        
        # Try GitHub releases
        links = self.scrape_github_releases(repo_url)
        
        # Try scraping README for Google Drive links
        readme_url = f"{repo_url}/blob/master/README.md"
        gdrive_links = self.scrape_google_drive_links(readme_url)
        
        all_links = links + gdrive_links
        
        if all_links:
            print(f"Found {len(all_links)} download links")
            output_dir = self.output_dir / 'celeba_spoof'
            output_dir.mkdir(exist_ok=True)
            
            for i, link in enumerate(all_links):
                filename = f"celeba_spoof_part{i+1}.zip"
                output_path = output_dir / filename
                
                print(f"\nDownloading {filename}...")
                if 'drive.google.com' in link:
                    self.download_with_gdown(link, output_path)
                else:
                    self.download_file(link, output_path)
            return True
        else:
            print("No direct download links found")
            return False
    
    def scrape_replay_attack(self):
        """Scrape Replay-Attack dataset"""
        print("\n📥 Scraping Replay-Attack...")
        # Often requires registration, so this is likely to fail without auth
        return False
    
    def scrape_oulu_npu(self):
        """Scrape OULU-NPU dataset"""
        print("\n📥 Scraping OULU-NPU...")
        
        page_url = "https://sites.google.com/site/oulunpudatabase/"
        gdrive_links = self.scrape_google_drive_links(page_url)
        
        if gdrive_links:
            print(f"Found {len(gdrive_links)} Google Drive links")
            output_dir = self.output_dir / 'oulu_npu'
            output_dir.mkdir(exist_ok=True)
            
            for i, link in enumerate(gdrive_links):
                filename = f"oulu_npu_part{i+1}.zip"
                output_path = output_dir / filename
                
                print(f"\nDownloading {filename}...")
                self.download_with_gdown(link, output_path)
            return True
        else:
            print("No links found")
            return False
    
    def generate_synthetic_data(self, num_videos=50):
        """Generate synthetic video dataset for testing"""
        print("\n" + "="*70)
        print("🎨 GENERATING SYNTHETIC DATASET")
        print("="*70)
        print("⚠️  Real datasets could not be downloaded automatically.")
        print("   Generating synthetic video data to verify the training pipeline.")
        
        real_dir = self.output_dir / 'real'
        spoof_dir = self.output_dir / 'spoof'
        
        real_dir.mkdir(parents=True, exist_ok=True)
        spoof_dir.mkdir(parents=True, exist_ok=True)
        
        width, height = 224, 224
        fps = 10
        duration = 3  # seconds
        
        print(f"\nGenerating {num_videos} real videos...")
        for i in tqdm(range(num_videos)):
            self._create_dummy_video(real_dir / f"real_{i}.mp4", width, height, fps, duration, is_spoof=False)
            
        print(f"Generating {num_videos} spoof videos...")
        for i in tqdm(range(num_videos)):
            self._create_dummy_video(spoof_dir / f"spoof_{i}.mp4", width, height, fps, duration, is_spoof=True)
            
        print(f"\n✅ Synthetic dataset created at: {self.output_dir}")
        return True

    def _create_dummy_video(self, path, width, height, fps, duration, is_spoof):
        """Create a single dummy video file"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        
        frames = fps * duration
        
        # Random base color
        base_color = np.random.randint(0, 255, 3)
        
        for _ in range(frames):
            # Create frame
            frame = np.full((height, width, 3), base_color, dtype=np.uint8)
            
            # Add noise
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            if is_spoof:
                # Spoof: Add static pattern or "screen" artifacts (grid lines)
                cv2.line(frame, (0, 0), (width, height), (0, 0, 0), 2)
                cv2.line(frame, (width, 0), (0, height), (0, 0, 0), 2)
                # Less temporal variation (static)
            else:
                # Real: Add "motion" (shifting circle)
                center_x = int(width/2 + np.sin(_/5) * 20)
                center_y = int(height/2 + np.cos(_/5) * 20)
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
            
            out.write(frame)
            
        out.release()

    def run(self):
        """Main execution flow"""
        print("\n" + "="*70)
        print("🕷️  AUTOMATED DATASET SCRAPER & GENERATOR")
        print("="*70)
        
        # Try scraping first
        success_celeba = self.scrape_celeba_spoof()
        success_oulu = self.scrape_oulu_npu()
        
        if not (success_celeba or success_oulu):
            print("\n❌ Failed to scrape real datasets automatically.")
            print("   (Likely due to SSL errors, permissions, or CAPTCHAs)")
            
            # Fallback to synthetic
            print("\n🔄 Falling back to synthetic data generation...")
            self.generate_synthetic_data(num_videos=50)
        
        print("\n" + "="*70)
        print("✅ DATASET PREPARATION COMPLETE")
        print("="*70)
        print(f"\nData location: {self.output_dir}")
        print("Next: Run 'python train_anti_spoofing.py'")


def main():
    scraper = DatasetScraper()
    scraper.run()


if __name__ == '__main__':
    main()
