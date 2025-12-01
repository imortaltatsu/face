"""
Automated Anti-Spoofing Dataset Scraper

Scrapes dataset websites to extract direct download links and automatically
downloads available datasets. No manual intervention required.
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


class DatasetScraper:
    """Automated dataset scraper and downloader"""
    
    def __init__(self, output_dir='data/anti_spoofing'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_github_releases(self, repo_url):
        """Scrape GitHub repository for release downloads"""
        try:
            # Convert to API URL
            repo_path = urlparse(repo_url).path.strip('/')
            api_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
            
            response = self.session.get(api_url)
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
            response = self.session.get(page_url)
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
    
    def scrape_direct_links(self, page_url, extensions=['.zip', '.tar.gz', '.tar', '.rar']):
        """Extract direct download links from webpage"""
        try:
            response = self.session.get(page_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Make absolute URL
                abs_url = urljoin(page_url, href)
                
                # Check if it's a download link
                if any(abs_url.endswith(ext) for ext in extensions):
                    links.append(abs_url)
            
            return links
        except Exception as e:
            print(f"Failed to scrape {page_url}: {e}")
        return []
    
    def download_file(self, url, output_path):
        """Download file with progress bar"""
        try:
            response = self.session.get(url, stream=True, timeout=30)
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
            gdown.download(gdrive_url, str(output_path), quiet=False)
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
        else:
            print("No direct download links found")
            print(f"Visit manually: {repo_url}")
    
    def scrape_replay_attack(self):
        """Scrape Replay-Attack dataset"""
        print("\n📥 Scraping Replay-Attack...")
        
        page_url = "https://www.idiap.ch/en/scientific-research/data/replayattack"
        
        # Try to find download links
        links = self.scrape_direct_links(page_url)
        
        if links:
            print(f"Found {len(links)} download links")
            output_dir = self.output_dir / 'replay_attack'
            output_dir.mkdir(exist_ok=True)
            
            for link in links:
                filename = Path(urlparse(link).path).name
                output_path = output_dir / filename
                
                print(f"\nDownloading {filename}...")
                self.download_file(link, output_path)
        else:
            print("Dataset requires registration")
            print(f"Visit: {page_url}")
    
    def scrape_oulu_npu(self):
        """Scrape OULU-NPU dataset"""
        print("\n📥 Scraping OULU-NPU...")
        
        page_url = "https://sites.google.com/site/oulunpudatabase/"
        
        # Try to find Google Drive links
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
        else:
            print("Dataset requires EULA signature")
            print(f"Visit: {page_url}")
    
    def scrape_all(self):
        """Scrape all available datasets"""
        print("\n" + "="*70)
        print("🕷️  AUTOMATED DATASET SCRAPER")
        print("="*70)
        
        datasets = [
            ('CelebA-Spoof', self.scrape_celeba_spoof),
            ('Replay-Attack', self.scrape_replay_attack),
            ('OULU-NPU', self.scrape_oulu_npu),
        ]
        
        for name, scraper_func in datasets:
            try:
                scraper_func()
            except Exception as e:
                print(f"\n❌ Error scraping {name}: {e}")
        
        print("\n" + "="*70)
        print("✅ SCRAPING COMPLETE")
        print("="*70)
        print(f"\nDatasets saved to: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Extract downloaded archives")
        print("  2. Organize into train/test splits")
        print("  3. Run: python train_anti_spoofing.py")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape anti-spoofing datasets')
    parser.add_argument('--output', default='data/anti_spoofing', help='Output directory')
    parser.add_argument('--dataset', choices=['celeba', 'replay', 'oulu', 'all'], 
                       default='all', help='Dataset to scrape')
    args = parser.parse_args()
    
    scraper = DatasetScraper(output_dir=args.output)
    
    if args.dataset == 'celeba':
        scraper.scrape_celeba_spoof()
    elif args.dataset == 'replay':
        scraper.scrape_replay_attack()
    elif args.dataset == 'oulu':
        scraper.scrape_oulu_npu()
    else:
        scraper.scrape_all()


if __name__ == '__main__':
    main()
