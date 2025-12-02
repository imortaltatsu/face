import json
from pathlib import Path
import os

def test_json_parsing(json_path):
    print(f"Loading dataset from JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    sequences = {} # path -> label (1=live, 0=spoof)
    
    # Mock data_dir
    data_dir = Path('data/video_liveness')
    dataset_root = data_dir
    if (dataset_root / 'CelebA_Spoof').exists():
        dataset_root = dataset_root / 'CelebA_Spoof'
        
    print(f"Dataset root: {dataset_root}")
    
    count = 0
    for file_rel_path, _ in data.items():
        # file_rel_path: Data/train/123/live/001.jpg
        path_parts = Path(file_rel_path).parts
        
        # Parent folder is the sequence
        parent_rel_path = Path(*path_parts[:-1])
        full_parent_path = dataset_root / parent_rel_path
        
        # Determine label from path
        is_live = 'live' in path_parts
        label = 1 if is_live else 0
        
        # Store unique sequences
        seq_path_str = str(full_parent_path)
        if seq_path_str not in sequences:
            sequences[seq_path_str] = label
            
        count += 1
        if count % 100000 == 0:
            print(f"Processed {count} files...", end='\r')
            
    print(f"\nGrouped into {len(sequences)} sequences")
    
    real_videos = []
    spoof_videos = []
    
    for path, label in sequences.items():
        if label == 1:
            real_videos.append(Path(path))
        else:
            spoof_videos.append(Path(path))
            
    print(f"Found {len(real_videos)} real sequences")
    print(f"Found {len(spoof_videos)} spoof sequences")
    
    if len(real_videos) > 0:
        print(f"Sample real: {real_videos[0]}")
    if len(spoof_videos) > 0:
        print(f"Sample spoof: {spoof_videos[0]}")

if __name__ == "__main__":
    if Path('train_label.json').exists():
        test_json_parsing('train_label.json')
    else:
        print("train_label.json not found")
