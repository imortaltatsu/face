#!/bin/bash
# Install rclone on Linux
echo "📦 Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

echo "✅ rclone installed."
echo "Now run 'rclone config' to set up your Google Drive remote."
