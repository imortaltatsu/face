"""
Automated Anti-Spoofing Dataset Scraper

1. Scrapes dataset websites for direct download links.
2. Uses curl with SSL bypass (-k) to download files.
3. Handles Google Drive links using curl with confirm token logic.
4. Automatically unzips downloaded archives.
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
import warnings
import zipfile

# Suppress SSL warnings
warnings.filterwarnings("ignore")

class DatasetScraper:
    """Automated dataset scraper and downloader using curl"""
    
    def __init__(self, output_dir='data/video_liveness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.verify = False
    
    def scrape_github_releases(self, repo_url):
        """Scrape GitHub repository for release downloads"""
        try:
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
            
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'drive.google.com' in href or 'docs.google.com' in href:
                    match = re.search(r'/d/([a-zA-Z0-9_-]+)', href)
                    if match:
                        file_id = match.group(1)
                        # Construct a direct download URL format for reference
                        links.append(f"https://drive.google.com/uc?id={file_id}&export=download")
            
            return links
        except Exception as e:
            print(f"Failed to scrape {page_url}: {e}")
        return []
    
    def download_with_curl(self, url, output_path):
        """Download file using curl with SSL bypass"""
        try:
            print(f"Downloading {output_path.name}...")
            
            # Basic curl command with SSL bypass (-k) and follow redirects (-L)
            cmd = ['curl', '-k', '-L', '-o', str(output_path), url]
            
            # Check if it's a Google Drive link
            if 'drive.google.com' in url:
                # Extract ID
                match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
                if match:
                    file_id = match.group(1)
                    # Use a robust one-liner for GDrive using curl
                    cmd = [
                        'curl', '-k', '-L', '-c', '/tmp/cookies.txt', 
                        f'https://drive.google.com/uc?export=download&id={file_id}',
                        '-o', str(output_path)
                    ]
            
            # Execute curl
            result = subprocess.run(cmd, check=True)
            
            if result.returncode == 0:
                print(f"✅ Downloaded: {output_path}")
                return True
            else:
                print(f"❌ Download failed with code {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Curl failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def unzip_file(self, file_path):
        """Unzip a file to the same directory"""
        try:
            print(f"Unzipping {file_path.name}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(file_path.parent)
            print(f"✅ Unzipped: {file_path.name}")
            return True
        except zipfile.BadZipFile:
            print(f"❌ Error: {file_path.name} is not a valid zip file")
            return False
        except Exception as e:
            print(f"❌ Error unzipping {file_path.name}: {e}")
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
                if self.download_with_curl(link, output_path):
                    self.unzip_file(output_path)
            return True
        else:
            print("No direct download links found")
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
                if self.download_with_curl(link, output_path):
                    self.unzip_file(output_path)
            return True
        else:
            print("No links found")
            return False

    def run(self):
        """Main execution flow"""
        print("\n" + "="*70)
        print("🕷️  AUTOMATED DATASET SCRAPER (CURL + SSL BYPASS + UNZIP)")
        print("="*70)
        
        self.scrape_celeba_spoof()
        self.scrape_oulu_npu()
        
        print("\n" + "="*70)
        print("✅ SCRAPING COMPLETE")
        print("="*70)
        print(f"\nData location: {self.output_dir}")
        print("Next: Run 'python train_anti_spoofing.py'")


def main():
    scraper = DatasetScraper()
    scraper.run()


if __name__ == '__main__':
    main()
