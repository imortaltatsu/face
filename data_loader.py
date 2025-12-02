import tensorflow as tf
import os
import json
import numpy as np
import cv2
from pathlib import Path

class VideoDataGenerator:
    """
    Generate video sequences for training using tf.data
    Supports loading from CelebA-Spoof JSON labels.
    """
    
    def __init__(self, data_dir, json_path=None, sequence_length=30, image_size=224, batch_size=16):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.real_videos = []
        self.spoof_videos = []
        
        if json_path:
            self._load_from_json(json_path)
        else:
            self._scan_directory()
            
    def _scan_directory(self):
        """Legacy method: Scan for "video" folders"""
        print("Scanning for image sequences (Directory Walk)...")
        # We look for the 'Data' folder which contains 'train' and 'test'
        search_path = self.data_dir / 'CelebA_Spoof' / 'Data'
        if not search_path.exists():
            search_path = self.data_dir
            
        print(f"Searching in: {search_path}")
        
        for root, dirs, files in os.walk(search_path):
            if 'live' in os.path.basename(root):
                if any(f.endswith('.png') or f.endswith('.jpg') for f in files):
                    self.real_videos.append(Path(root))
            elif 'spoof' in os.path.basename(root):
                if any(f.endswith('.png') or f.endswith('.jpg') for f in files):
                    self.spoof_videos.append(Path(root))
                    
        print(f"Found {len(self.real_videos)} real sequences")
        print(f"Found {len(self.spoof_videos)} spoof sequences")

    def _load_from_json(self, json_path):
        """Load sequences from CelebA-Spoof JSON label file"""
        print(f"Loading dataset from JSON: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Data format: {"Data/train/ID/type/img.jpg": [labels...], ...}
        # We need to group by parent folder to form sequences
        
        sequences = {} # path -> label (1=live, 0=spoof)
        
        # Adjust root path based on common structure
        dataset_root = self.data_dir
        if (dataset_root / 'CelebA_Spoof').exists():
            dataset_root = dataset_root / 'CelebA_Spoof'
            
        print(f"Dataset root: {dataset_root}")
        
        count = 0
        for file_rel_path, labels in data.items():
            # file_rel_path: Data/train/123/live/001.jpg
            path_parts = Path(file_rel_path).parts
            
            # Parent folder is the sequence
            # e.g. Data/train/123/live
            parent_rel_path = Path(*path_parts[:-1])
            full_parent_path = dataset_root / parent_rel_path
            
            # Determine label from JSON (Index 43 is live/spoof)
            # Verification showed: 0=Live, 1=Spoof
            # We want: 1=Live, 0=Spoof (for model consistency)
            if len(labels) > 43:
                json_label = labels[43]
                # Map: 0 (Live) -> 1, 1 (Spoof) -> 0
                label = 1 if json_label == 0 else 0
            else:
                # Fallback to path string if JSON is incomplete
                is_live = 'live' in path_parts
                label = 1 if is_live else 0
            
            # Store unique sequences
            # We use string path as key
            seq_path_str = str(full_parent_path)
            if seq_path_str not in sequences:
                sequences[seq_path_str] = label
                
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} files...", end='\r')
                
        print(f"\nGrouped into {len(sequences)} sequences")
        
        for path, label in sequences.items():
            if label == 1:
                self.real_videos.append(Path(path))
            else:
                self.spoof_videos.append(Path(path))
                
        print(f"Found {len(self.real_videos)} real sequences")
        print(f"Found {len(self.spoof_videos)} spoof sequences")
    
    def crop_face_from_bb(self, image, bb_path):
        """
        Crop face using bounding box from _BB.txt
        """
        if not os.path.exists(bb_path):
            return cv2.resize(image, (self.image_size, self.image_size))
            
        try:
            with open(bb_path, 'r') as f:
                content = f.read().strip()
                
                import re
                match = re.search(r'bbox\s*=\s*\[([\d\s\.]+)\]', content)
                if match:
                    vals = [float(x) for x in match.group(1).split()]
                else:
                    vals = [float(x) for x in content.split() if x.replace('.','',1).isdigit()]
                
                if len(vals) < 4:
                    return cv2.resize(image, (self.image_size, self.image_size))
                
                x_224, y_224, w_224, h_224 = vals[:4]
                
                # Real image dims
                real_h, real_w = image.shape[:2]
                
                # Scale coordinates
                x1 = int(x_224 * (real_w / 224.0))
                y1 = int(y_224 * (real_h / 224.0))
                w1 = int(w_224 * (real_w / 224.0))
                h1 = int(h_224 * (real_h / 224.0))
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                w1 = min(real_w, w1)
                h1 = min(real_h, h1)
                
                # Ensure valid crop
                if w1 <= 0 or h1 <= 0:
                    return cv2.resize(image, (self.image_size, self.image_size))
                
                face = image[y1:y1+h1, x1:x1+w1]
                
                # Resize to model input
                face = cv2.resize(face, (self.image_size, self.image_size))
                return face
                
        except Exception as e:
            return cv2.resize(image, (self.image_size, self.image_size))

    def _get_sequence_paths_wrapper(self, folder_path_tensor, label):
        """
        Python wrapper to list files in a folder.
        Returns: (image_paths, bb_paths, label)
        """
        folder_path = folder_path_tensor.numpy().decode('utf-8')
        folder = Path(folder_path)
        
        # Get all images sorted (support png and jpg)
        images = sorted(list(folder.glob('*.png')) + list(folder.glob('*.jpg')))
        
        if not images:
            return [], [], label
            
        # Sampling logic
        if len(images) < self.sequence_length:
            # Pad by repeating last frame
            indices = np.linspace(0, len(images) - 1, len(images), dtype=int)
            while len(indices) < self.sequence_length:
                indices = np.concatenate([indices, indices])
            indices = indices[:self.sequence_length]
        else:
            # Uniform sample
            indices = np.linspace(0, len(images) - 1, self.sequence_length, dtype=int)
            
        selected_imgs = [str(images[i]) for i in indices]
        # BB paths: replace extension with _BB.txt
        selected_bbs = [str(p).rsplit('.', 1)[0] + '_BB.txt' for p in selected_imgs]
        
        return selected_imgs, selected_bbs, label

    def _load_and_process_sequence(self, image_paths, bb_paths, label):
        """
        TensorFlow-native loading and processing (Runs in C++ threads, No GIL)
        Input: Tensor of string paths
        """
        
        def process_frame(img_path, bb_path):
            # 1. Read Image
            img_content = tf.io.read_file(img_path)
            img = tf.image.decode_png(img_content, channels=3)
            img = tf.cast(img, tf.float32)
            
            # Get dimensions
            shape = tf.shape(img)
            real_h = tf.cast(shape[0], tf.float32)
            real_w = tf.cast(shape[1], tf.float32)
            
            # 2. Read BB (If exists)
            bb_content = tf.io.read_file(bb_path)
            
            # Parse BB: "bbox = [x y w h score]"
            bb_text = tf.strings.regex_replace(bb_content, "[^0-9. ]", " ")
            bb_text = tf.strings.strip(bb_text)
            bb_vals = tf.strings.to_number(tf.strings.split(bb_text), out_type=tf.float32)
            
            def crop_face():
                x_224 = bb_vals[0]
                y_224 = bb_vals[1]
                w_224 = bb_vals[2]
                h_224 = bb_vals[3]
                
                # Scale to real image
                scale_x = real_w / 224.0
                scale_y = real_h / 224.0
                
                x = x_224 * scale_x
                y = y_224 * scale_y
                w = w_224 * scale_x
                h = h_224 * scale_y
                
                # Convert to int
                x = tf.cast(x, tf.int32)
                y = tf.cast(y, tf.int32)
                w = tf.cast(w, tf.int32)
                h = tf.cast(h, tf.int32)
                
                # Clip
                x = tf.maximum(0, x)
                y = tf.maximum(0, y)
                w = tf.minimum(tf.cast(real_w, tf.int32) - x, w)
                h = tf.minimum(tf.cast(real_h, tf.int32) - y, h)
                
                cropped = tf.image.crop_to_bounding_box(img, y, x, h, w)
                return tf.image.resize(cropped, [self.image_size, self.image_size])

            def full_image():
                return tf.image.resize(img, [self.image_size, self.image_size])
                
            # Check if we have enough values
            has_bb = tf.size(bb_vals) >= 4
            processed_img = tf.cond(has_bb, crop_face, full_image)
            
            # Normalize
            return processed_img / 255.0

        # Map process_frame over the sequence of paths
        frames = tf.map_fn(
            lambda x: process_frame(x[0], x[1]), 
            elems=(image_paths, bb_paths),
            fn_output_signature=tf.float32
        )
        
        return frames, label

    def _create_tf_dataset(self, video_list, is_training=True):
        """Create optimized tf.data.Dataset"""
        if not video_list:
            return None
            
        # Unzip to separate lists
        video_paths, labels = zip(*video_list)
        
        # 1. Dataset of folder paths (Lightweight)
        dataset = tf.data.Dataset.from_tensor_slices((list(video_paths), list(labels)))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(video_paths))
        
        # 2. Python Stage: Get file paths (Fast, Metadata only)
        dataset = dataset.map(
            lambda path, label: tf.py_function(
                self._get_sequence_paths_wrapper, 
                inp=[path, label], 
                Tout=[tf.string, tf.string, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 3. Filter empty sequences
        dataset = dataset.filter(lambda imgs, bbs, lbl: tf.size(imgs) > 0)
        
        # 4. TensorFlow Stage: Read & Decode (Heavy, Parallel, No GIL)
        dataset = dataset.map(
            self._load_and_process_sequence,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 5. Set Shapes
        def set_shapes(frames, label):
            frames.set_shape((self.sequence_length, self.image_size, self.image_size, 3))
            label.set_shape([])
            return frames, label
            
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def create_dataset(self, validation_split=0.2):
        """Create tf.data.Dataset for training with parallel loading"""
        all_videos = [(str(v), 1) for v in self.real_videos] + [(str(v), 0) for v in self.spoof_videos]
        np.random.shuffle(all_videos)
        
        if not all_videos:
            print("❌ No sequences found! Check dataset path.")
            return None, None
        
        split_idx = int(len(all_videos) * (1 - validation_split))
        train_videos = all_videos[:split_idx]
        val_videos = all_videos[split_idx:]
        
        print(f"Training sequences: {len(train_videos)}")
        print(f"Validation sequences: {len(val_videos)}")
        
        train_dataset = self._create_tf_dataset(train_videos, is_training=True)
        val_dataset = self._create_tf_dataset(val_videos, is_training=False)
        
        return train_dataset, val_dataset

    def get_dataset(self, is_training=True):
        """Create a single tf.data.Dataset from all loaded videos"""
        all_videos = [(str(v), 1) for v in self.real_videos] + [(str(v), 0) for v in self.spoof_videos]
        
        if is_training:
            np.random.shuffle(all_videos)
            
        print(f"Total sequences: {len(all_videos)}")
        return self._create_tf_dataset(all_videos, is_training=is_training)
