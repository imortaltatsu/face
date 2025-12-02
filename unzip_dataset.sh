#!/bin/bash
# Install 7-zip and extract CelebA-Spoof

echo "📦 Installing 7-zip..."
if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get update
    sudo apt-get install -y p7zip-full
elif [ -x "$(command -v yum)" ]; then
    sudo yum install -y p7zip
elif [ -x "$(command -v brew)" ]; then
    brew install p7zip
else
    echo "❌ Could not detect package manager. Please install 7-zip manually."
    exit 1
fi

echo "✅ 7-zip installed."

DATA_DIR="data/video_liveness/celeba_spoof"
cd "$DATA_DIR" || exit

echo "📂 Extracting CelebA-Spoof dataset..."
# 7z x automatically handles split archives (001, 002, etc) if you point to the first one
if [ -f "CelebA_Spoof.zip.001" ]; then
    7z x CelebA_Spoof.zip.001 -y
    echo "✅ Extraction complete."
else
    echo "❌ CelebA_Spoof.zip.001 not found in $DATA_DIR"
fi
