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
        
        # Complete list of 74 File IDs for CelebA-Spoof dataset parts
        # Extracted directly from the Google Drive folder + User provided links
        self.file_ids = [
            '1-UYx3LoCloTNchmSq5otd4GxkWVTEGs_', '11EdKR_DrmAmbvIXhPI6bj7FfMGNh0lAh', '124cQ_o0MwY3jsieIEHqr2mfHpO3BuKrE',
            '14d1TYmcVXg1TlmuywM0HgOKcPqa71Rmq', '15n2kpdRqu5rhyIYWIy50JleUHhEXezxt', '16VogXk2Onsva5i9wszhTL4Sigg-kFlQN',
            '16gXdRiWd1tTiuRe5beaVYGXAUhX2m5f_', '18qd2y1fBPiqQqgSM-vrQq2rMbXeSHWLy', '18z-KimWHTh6-T05KzVlENogci5NF-PQH',
            '1Ac-DvMyCeUzRYsT1R_NrK1zcsQlRy_QL', '1BQtBSRoA0jFxt7zS9sZgJUFkZUQYj_rU', '1BTtMkRJO-DOhrZhR0kjenegN41mM5qbU',
            '1E6zkuosSUPAQBVGSWg5FlqnhhlACCeYS', '1EmVAkbAf3ZfGfVZrZpb7jHqsWqIy3wJa', '1F-FI0JO8GLfA2QF5X8SSnjMpNeV5avZH',
            '1FO7e7SBv50VOPaQJb_Kh9QfUHXwBrqXb', '1FZ4OGHJ_CWCf04oysgmCSONRTTYNsf5Y', '1FhekjS7l-0pDsymm_GM9aMJhh8VwcEah',
            '1G16gR7eM45wAaPt8jVFdDt8Dx510QxNY', '1HgwT6-NLplcBkowdSHQz5VZEGEsFFDRd', '1IjmswQCGQEtpdO3jr9TyJSNj5ikhOCDA',
            '1JPAB1jjLVdn8BBJk4hydDZq-IV5czrGq', '1JTGaAL0d6QhwDoY1FRGq_IoexBXKCDAh', '1JsY9XCjW_Du5VZyPnkuh1Qdajs5aGopt',
            '1K-6UwtmUvrW7sZqaUvGYIkqpPC1G3K1V', '1MOfdHWaU_ijqmDjLE34KG5AgmnBuTUnC', '1MZnG9Uzf5q86deEjev4REeOevKsN4eyy',
            '1MiHlWChmvOy53eTyiwWvd6hRV8zSeEYJ', '1NO1vsdldaVj2pe93QdNfBZ3Ao6VWYvM7', '1Nkwv2phRRdbEA2Ief1Y9MCrd7UvZIs-G',
            '1NxlMSgvJSlDJMRFIfq_cT_RRybYS8eNW', '1OsOiow42GS4wbE7o1csPA2HtKLSgXT5_', '1Oy-5J10hsrZ2gyHH9U2oet0F92oaEqSq',
            '1P7GD4fTK36dBvASLEVkeG01CUoDbX_Yl', '1Qpsou4EJJVd8aVyr9UKdtNvU_Rap3fAU', '1U34bffAOUmBsR5YSF_B6Nz2k-97rMFTM',
            '1VJoKI6vJJ5lb-JRazmjhDv_NsXzVpQTx', '1VX9YmcjCAWXYeQ3OjbEwO7cCPpn0UUzd', '1W-IQReuw1Rc6PfTeLpeDxu-1v3n5tRcl',
            '1WqikxT7XCMZLaPaYfOo0gYZOxeqAiFcP', '1X2Bg5acR0sVkrR4_h3BEMnm6PQFlwzVW', '1XZ5oYVAO8J8Zjp_dHb2-4YHWhobE9r2s',
            '1XwEjdvaBG_xFLVxauZK5MoeeFAnygkYI', '1YI5f9Er6TGq3nlmut2zUfC-bumPtZUOg', '1_07Q7VA4-4R5fvHpkyoorhD5igeathko',
            '1_pA7wv_nMgla0d3KH2cqN8JizL7vZyxP', '1canVFU2jxJKs6VbcsvHPQeM1qEa8UVwY', '1dJ3m7aygdJhcweTmPkZPUxFIYbAM6ysM',
            '1dihf4baq-_17g2tj1K3ya4Nf7xxMC96O', '1eoLXTSOr6MF53D9cElrkg7o5l7yI0xde', '1euzrQ4qxGLpOgmW1tIujvDUSdZFlsWmn',
            '1fsS9it_vKV9bgntj8awTx-mRE1uhN9cA', '1g9Ucb8m3l8rV5IpVG1Y2VTkNcZz8DImF', '1gd0M4qpxzqrnOkCl7hv7vnCWV4JJTFwM',
            '1iswrU0QuYSbzwKk0g0TWDD52hbtOps21', '1j44juO753d5cp7xLA9ul5bTN8upjgrjY', '1jZerN3JzvTjxucNnQExqq61gDjxS0l5v',
            '1jxuF9VFgRwD7ayd3c19cxPb7zLLFM1fp', '1krpR7L20uRnOgNo1gx59pOhEcCa_pkyE', '1lU_RQN_HtB_8yYbRwkCzapfc5-4d0gW1',
            '1lzd-nWagDEqldxkVFRK5-dwGqIf8XYp7', '1m9dVjwU-KajHbuUMVHgENg62fUdCZg1f', '1mpMTb5ODq-9NV4IEVFwEDhUbYSNfVQtC',
            '1mqXdA81AaAMM-MZvu6h0vCFh0gXWoHNU', '1nKmKMvWV4kqeIPKQkoYnLPGB2QHFS7U1', '1oftTxjUVLJdEOGADJzXjVNlUM6_BLlaO',
            '1s11ngviLIOiZcWjX_mkjEdQc7H_oY52-', '1st5Yh8yRQGAmI02iueWPm2rrJN1TpOlC', '1v7_ruZ2xvLCl9yQp9dEoPs94qe-rNDjL',
            '1wNgEJSBlZuQKePCTbwLM_9jBuIg3Hj1f', '1wt9mOElhnkoYzlVPm5N_69SFhLSvy3YA', '1xOq1PXanKDpUs1-JAcNaz6Hs5I8MFxxn',
            '1ychf3AazgFb73Z1LIJ_lLhQwvYnOBCXd', '1z7c31kyQdrFxbpT6ZblIwAKRnJ9c2MTf'
        ]
        
    def download_files(self, proxy=None):
        """Download files individually using gdown with proxy rotation"""
        import gdown
        
        print("\n" + "="*70)
        print("📥 Downloading CelebA-Spoof Dataset Parts...")
        print("="*70)
        print(f"Target Directory: {self.celeba_dir}")
        print(f"Total Files: {len(self.file_ids)}")
        
        # Proxies to use (User provided + CLI argument)
        proxies = []
        if proxy:
            proxies.append(proxy)
        
        # Add hardcoded proxies from user
        proxies.extend([
            "http://84.17.47.150:9002",
            "http://84.17.47.149:9002"
        ])
        
        # Ensure we have unique proxies
        proxies = list(set(proxies))
        
        if proxies:
            print(f"🌐 Loaded {len(proxies)} proxies for rotation.")
        
        success_count = 0
        
        for i, file_id in enumerate(self.file_ids):
            # Predict filename based on index (001, 002, ...)
            expected_filename = f"CelebA_Spoof.zip.{i+1:03d}"
            expected_path = self.celeba_dir / expected_filename
            
            if expected_path.exists():
                print(f"[{i+1}/{len(self.file_ids)}] Skipping {expected_filename} (already exists)")
                success_count += 1
                continue

            print(f"\n[{i+1}/{len(self.file_ids)}] Downloading file ID: {file_id}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Try direct download first, then proxies
            download_attempts = [None] + proxies
            file_downloaded = False
            
            for attempt_idx, current_proxy in enumerate(download_attempts):
                proxy_msg = f"via Proxy: {current_proxy}" if current_proxy else "Direct"
                if attempt_idx > 0:
                    print(f"   Retrying {proxy_msg}...")
                
                try:
                    # Use gdown to download
                    # verify=False to bypass SSL errors if any
                    output_path = gdown.download(url, output=str(expected_path), quiet=False, verify=False, proxy=current_proxy)
                    
                    if output_path:
                        # gdown with output arg returns the path
                        file_downloaded = True
                        success_count += 1
                        break # Success, move to next file
                    else:
                        print(f"   ❌ Download failed {proxy_msg} (no output path)")
                        
                except Exception as e:
                    print(f"   ❌ Error {proxy_msg}: {e}")
                    # Continue to next proxy
            
            if not file_downloaded:
                print(f"   ❌ Failed to download {file_id} after trying all options.")
                print("      (You may need to wait 24h if this is a quota error on all IPs)")
                
        print(f"\n✅ Downloaded/Found {success_count}/{len(self.file_ids)} files.")
        return success_count == len(self.file_ids) # Only return True if ALL files are present

    def unzip_files(self):
        """Unzip all zip files in the directory, handling split archives"""
        print(f"\n📦 Checking for zip files to extract in {self.celeba_dir}...")
        
        # Check for split files (CelebA_Spoof.zip.001, .002, etc)
        split_files = sorted(list(self.celeba_dir.glob('CelebA_Spoof.zip.*')))
        
        if split_files:
            print(f"   Found {len(split_files)} split zip parts.")
            full_zip = self.celeba_dir / 'CelebA_Spoof_Full.zip'
            
            if not full_zip.exists():
                print("   🔗 Combining available split files into single archive...")
                # Use cat to combine efficiently
                cmd = f"cat {self.celeba_dir}/CelebA_Spoof.zip.* > {full_zip}"
                try:
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"   ✅ Combined to {full_zip}")
                except Exception as e:
                    print(f"   ❌ Failed to combine files: {e}")
                    return
            else:
                print("   ℹ️  Combined archive already exists.")

            print(f"   🔓 Unzipping {full_zip}...")
            try:
                # Try standard unzip first
                subprocess.run(['unzip', '-o', str(full_zip), '-d', str(self.celeba_dir)], check=True)
                print(f"   ✅ Extracted successfully.")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  Standard unzip failed (likely due to missing parts).")
                print(f"   🔧 Attempting to repair archive with 'zip -FF'...")
                
                repaired_zip = self.celeba_dir / 'CelebA_Spoof_Repaired.zip'
                try:
                    # zip -FF (fix fix) scans the file and reconstructs the central directory
                    subprocess.run(['zip', '-FF', str(full_zip), '--out', str(repaired_zip)], check=True)
                    print(f"   ✅ Repaired archive created: {repaired_zip}")
                    
                    print(f"   🔓 Unzipping repaired archive...")
                    subprocess.run(['unzip', '-o', str(repaired_zip), '-d', str(self.celeba_dir)], check=True)
                    print(f"   ✅ Extracted repaired archive successfully.")
                except Exception as e:
                    print(f"   ❌ Repair/Extraction failed: {e}")
                    print("   💡 Suggestion: Try installing '7zip' and running: 7z x CelebA_Spoof.zip.001")
                
        else:
            # Standard unzip for non-split files
            zip_files = sorted(list(self.celeba_dir.glob('*.zip')))
            if not zip_files:
                print("   No zip files found.")
                return

            for zip_path in zip_files:
                if zip_path.name == 'CelebA_Spoof_Full.zip': continue # Skip the combined one if we just made it
                
                print(f"   Unzipping {zip_path.name}...")
                try:
                    subprocess.run(['unzip', '-o', str(zip_path), '-d', str(zip_path.parent)], check=True)
                    print(f"   ✅ Extracted")
                except Exception as e:
                    print(f"   ❌ Error: {e}")

    def run(self, proxy=None):
        """Main execution flow"""
        install_gdown()
        
        if self.download_files(proxy=proxy):
            self.unzip_files()
            print("\n" + "="*70)
            print("✅ DATASET PREPARATION COMPLETE")
            print("="*70)
        else:
            print("\n❌ Dataset preparation failed (or partial).")
            # Attempt unzip even if partial, as we have repair logic now
            self.unzip_files()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download CelebA-Spoof Dataset')
    parser.add_argument('--proxy', type=str, help='Proxy URL (e.g., http://user:pass@host:port)')
    args = parser.parse_args()

    downloader = DatasetDownloader()
    downloader.run(proxy=args.proxy)


if __name__ == '__main__':
    main()
