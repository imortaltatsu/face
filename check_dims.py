
import cv2
import os
import glob

def check_dims():
    # Find a png file
    files = glob.glob("data/video_liveness/celeba_spoof/CelebA_Spoof/Data/**/*.png", recursive=True)
    if not files:
        print("No PNG files found.")
        return

    img_path = files[0]
    img = cv2.imread(img_path)
    print(f"Image: {img_path}")
    print(f"Dimensions: {img.shape}")
    
    # Check for BB file
    bb_path = img_path.replace(".png", "_BB.txt")
    if os.path.exists(bb_path):
        with open(bb_path, 'r') as f:
            print(f"BB Content: {f.read()}")
    else:
        print("BB file not found.")

if __name__ == "__main__":
    check_dims()
