import json
from pathlib import Path

def verify_labels():
    json_path = 'train_label.json'
    if not Path(json_path).exists():
        print("train_label.json not found")
        return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")
    
    # Check correlation between index 43 and path 'live'/'spoof'
    live_path_vals = []
    spoof_path_vals = []
    
    mismatch_count = 0
    
    for path, labels in data.items():
        if len(labels) <= 43:
            continue
            
        val = labels[43]
        
        if 'live' in path:
            live_path_vals.append(val)
            if val != 0: # Hypothesizing 0 is live? Or 1?
                pass
        elif 'spoof' in path:
            spoof_path_vals.append(val)
            
    from collections import Counter
    print("Values for paths containing 'live':")
    print(Counter(live_path_vals))
    
    print("Values for paths containing 'spoof':")
    print(Counter(spoof_path_vals))

if __name__ == "__main__":
    verify_labels()
