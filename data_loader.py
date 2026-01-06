import tensorflow as tf
import os
import json
import numpy as np
import cv2
from pathlib import Path

class FaceImageDataGenerator:
    """
    Generate individual face images for training using tf.data
    Supports loading from CelebA-Spoof JSON labels.
    """
    
    def __init__(self, data_dir, json_path=None, image_size=224, batch_size=32):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # List of (image_path, label) tuples
        self.real_images = []
        self.spoof_images = []
        
        if json_path:
            self._load_from_json(json_path)
        else:
            self._scan_directory()
            
    def _scan_directory(self):
        """Legacy method: Scan for images in folders"""
        print("Scanning for images (Directory Walk)...")
        search_path = self.data_dir / 'CelebA_Spoof' / 'Data'
        if not search_path.exists():
            search_path = self.data_dir
            
        print(f"Searching in: {search_path}")
        
        for root, dirs, files in os.walk(search_path):
            if 'live' in os.path.basename(root):
                for f in files:
                    if f.endswith('.png') or f.endswith('.jpg'):
                        self.real_images.append(str(Path(root) / f))
            elif 'spoof' in os.path.basename(root):
                for f in files:
                    if f.endswith('.png') or f.endswith('.jpg'):
                        self.spoof_images.append(str(Path(root) / f))
                    
        print(f"Found {len(self.real_images)} real images")
        print(f"Found {len(self.spoof_images)} spoof images")

    def _load_from_json(self, json_path):
        """Load images from CelebA-Spoof JSON label file"""
        print(f"Loading dataset from JSON: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Data format: {"Data/train/ID/type/img.jpg": [labels...], ...}
        
        # Adjust root path based on common structure
        dataset_root = self.data_dir
        if (dataset_root / 'CelebA_Spoof').exists():
            dataset_root = dataset_root / 'CelebA_Spoof'
        elif (dataset_root / 'celeba_spoof' / 'CelebA_Spoof').exists():
            # Server layout keeps CelebA under a nested "celeba_spoof" folder
            dataset_root = dataset_root / 'celeba_spoof' / 'CelebA_Spoof'
            
        print(f"Dataset root: {dataset_root}")
        
        count = 0
        for file_rel_path, labels in data.items():
            # file_rel_path: Data/train/123/live/001.jpg
            full_path = dataset_root / file_rel_path
            
            # Determine label from JSON (Index 43 is live/spoof)
            # Verification showed: 0=Live, 1=Spoof
            # We want: 1=Live, 0=Spoof (for model consistency)
            if len(labels) > 43:
                json_label = labels[43]
                # Map: 0 (Live) -> 1, 1 (Spoof) -> 0
                label = 1 if json_label == 0 else 0
            else:
                # Fallback to path string if JSON is incomplete
                is_live = 'live' in str(file_rel_path)
                label = 1 if is_live else 0
            
            # Store image path and label
            if label == 1:
                self.real_images.append(str(full_path))
            else:
                self.spoof_images.append(str(full_path))
                
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} files...", end='\r')
                
        print(f"\nFound {len(self.real_images)} real images")
        print(f"Found {len(self.spoof_images)} spoof images")
    
    def _load_and_process_image(self, img_path, label):
        """
        TensorFlow-native loading and processing
        """
        # 1. Read Image
        img_content = tf.io.read_file(img_path)
        # Decode both PNG and JPG
        img = tf.image.decode_image(img_content, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        
        # Get dimensions
        shape = tf.shape(img)
        real_h = tf.cast(shape[0], tf.float32)
        real_w = tf.cast(shape[1], tf.float32)
        
        # 2. Read BB (If exists)
        # Construct BB path: replace extension with _BB.txt
        # Note: This string manipulation in TF is tricky if extensions vary.
        # We'll assume standard naming convention.
        
        # Regex replace extension with _BB.txt
        # We need to handle both .jpg and .png
        bb_path = tf.strings.regex_replace(img_path, r"\.(png|jpg|jpeg)$", "_BB.txt")
        
        # Try to read BB file
        # Since tf.io.read_file fails if file missing, we need a safe way.
        # But we can't check existence easily in graph mode without py_function.
        # However, for efficiency, we can try to rely on the fact that dataset should have BBs.
        # If not, we might need a py_function check.
        # Let's use a py_function for the BB path check to be safe, 
        # or just skip BB cropping if we want pure TF speed and assume full image.
        # Given the dataset has BBs, let's try to use them.
        
        # To be robust and fast, let's just resize the full image for now.
        # Cropping in graph mode with optional files is complex.
        # If we really need BBs, we should pre-process or use py_function.
        # Let's use full image resize for simplicity and speed in this "Image Model" phase.
        # If accuracy is low, we can add BB cropping back via py_function.
        
        img = tf.image.resize(img, [self.image_size, self.image_size])
        
        # Normalize
        img = img / 255.0
        
        return img, label

    def _create_tf_dataset(self, image_list, label_list, is_training=True):
        """Create optimized tf.data.Dataset"""
        if not image_list:
            return None
            
        # 1. Dataset of paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(image_list), 100000))
        
        # 2. Map loading function
        # Using pure TF ops for speed where possible
        dataset = dataset.map(
            self._load_and_process_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 3. Set Shapes
        def set_shapes(img, label):
            img.set_shape((self.image_size, self.image_size, 3))
            label.set_shape([])
            return img, label
            
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def create_dataset(self, validation_split=0.2):
        """Create tf.data.Dataset for training"""
        # Combine lists
        all_images = self.real_images + self.spoof_images
        all_labels = [1] * len(self.real_images) + [0] * len(self.spoof_images)
        
        # Shuffle together
        combined = list(zip(all_images, all_labels))
        np.random.shuffle(combined)
        
        if not combined:
            print("❌ No images found! Check dataset path.")
            return None, None
            
        all_images, all_labels = zip(*combined)
        all_images = list(all_images)
        all_labels = list(all_labels)
        
        # Split train/val
        split_idx = int(len(all_images) * (1 - validation_split))
        
        train_imgs = all_images[:split_idx]
        train_lbls = all_labels[:split_idx]
        
        val_imgs = all_images[split_idx:]
        val_lbls = all_labels[split_idx:]
        
        print(f"Training images: {len(train_imgs)}")
        print(f"Validation images: {len(val_imgs)}")
        
        train_dataset = self._create_tf_dataset(train_imgs, train_lbls, is_training=True)
        val_dataset = self._create_tf_dataset(val_imgs, val_lbls, is_training=False)
        
        return train_dataset, val_dataset

    def get_dataset(self, is_training=True):
        """Create a single tf.data.Dataset from all loaded images"""
        all_images = self.real_images + self.spoof_images
        all_labels = [1] * len(self.real_images) + [0] * len(self.spoof_images)
        
        combined = list(zip(all_images, all_labels))
        if is_training:
            np.random.shuffle(combined)
            
        all_images, all_labels = zip(*combined)
        
        print(f"Total images: {len(all_images)}")
        return self._create_tf_dataset(list(all_images), list(all_labels), is_training=is_training)
