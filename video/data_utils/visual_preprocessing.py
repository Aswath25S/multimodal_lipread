"""
Visual preprocessing module for GLips AVSR project.
Uses MediaPipe Face Mesh to extract lip regions from video frames.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import config

sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/video')
from config.config import load_config


class LipRegionExtractor:
    """
    Extract lip regions from video frames using MediaPipe Face Mesh.
    """
    
    def __init__(self, target_size=(44, 44), padding_mode='average'):
        """
        Initialize the LipRegionExtractor.
        
        Args:
            target_size (tuple): Target size for lip region images (height, width)
            padding_mode (str): Mode for padding ('average', 'zeros', 'border')
        """
        self.target_size = target_size
        self.padding_mode = padding_mode
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Assuming one face per video
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (upper and lower lips)
        # MediaPipe Face Mesh uses 468 landmarks
        # Lips are roughly from indices 60-72 (upper outer lip), 
        # 96-103 (lower outer lip), 76-88 (upper inner lip), 
        # 89-95 (lower inner lip)
        # self.lip_landmark_indices = list(range(60, 73)) + list(range(96, 104)) + \
        #                            list(range(76, 89)) + list(range(89, 96))
        self.lip_landmark_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # outer
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308    # inner
            ]
    
    def extract_lip_region(self, frame):
        """
        Extract lip region from a single frame.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            np.ndarray: Extracted lip region image or None if no face is detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract lip landmarks
        h, w, _ = frame.shape
        lip_landmarks = []
        for idx in self.lip_landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            lip_landmarks.append((x, y))
        
        # Calculate bounding box of lip region
        x_min = min(x for x, y in lip_landmarks)
        x_max = max(x for x, y in lip_landmarks)
        y_min = min(y for x, y in lip_landmarks)
        y_max = max(y for x, y in lip_landmarks)
        
        # Add a margin (20% of lip height/width)
        height = y_max - y_min
        width = x_max - x_min
        margin_h = int(height * 0.4)
        margin_w = int(width * 0.4)
        
        y_min = max(0, y_min - margin_h)
        y_max = min(h, y_max + margin_h)
        x_min = max(0, x_min - margin_w)
        x_max = min(w, x_max + margin_w)
        
        # Extract lip region
        lip_region = frame[y_min:y_max, x_min:x_max]
        
        # Resize and pad to target size
        return self._resize_and_pad(lip_region)
    
    def _resize_and_pad(self, image):
        """
        Resize and pad the lip region image to the target size.
        
        Args:
            image (np.ndarray): Input lip region image
            
        Returns:
            np.ndarray: Resized and padded image
        """
        if image is None or image.size == 0:
            # If no image is provided, return a blank image
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        h, w, c = image.shape
        target_h, target_w = self.target_size
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h
        
        # Resize the image while maintaining aspect ratio
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            # Image is taller than target
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Create a blank canvas
        if self.padding_mode == 'average':
            # Use average pixel value for padding
            avg_color = np.mean(resized_image, axis=(0, 1)).astype(np.uint8)
            padded_image = np.full((target_h, target_w, c), avg_color, dtype=np.uint8)
        elif self.padding_mode == 'zeros':
            # Use zeros for padding
            padded_image = np.zeros((target_h, target_w, c), dtype=np.uint8)
        else:  # border padding (default)
            padded_image = np.zeros((target_h, target_w, c), dtype=np.uint8)
        
        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place the resized image on the canvas
        padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image
        
        return padded_image
    
    def extract_lip_sequence(self, video_path, num_frames=29):
        """
        Extract a sequence of lip regions from a video.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract
            
        Returns:
            np.ndarray: Sequence of lip region images with shape (num_frames, height, width, channels)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= num_frames:
            # If video has fewer frames than required, duplicate frames
            frame_indices = np.arange(total_frames)
            frame_indices = np.append(frame_indices, [frame_indices[-1]] * (num_frames - total_frames))
        else:
            # Otherwise, sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Extract lip regions from selected frames
        lip_sequence = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame couldn't be read, add a blank frame
                lip_sequence.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
                continue
            
            lip_region = self.extract_lip_region(frame)
            
            if lip_region is None:
                # If no lips detected, use a blank frame
                lip_region = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            
            lip_sequence.append(lip_region)
        
        cap.release()
        
        return np.array(lip_sequence)
    
    def close(self):
        """
        Release MediaPipe resources.
        """
        self.face_mesh.close()


def process_dataset(config_path):
    """
    Process the GLips dataset to extract lip regions from all videos.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get dataset path
    dataset_path = config.get('dataset.root_dir')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Get preprocessing configuration
    image_size = config.get('preprocessing.image_size')
    target_size = (image_size[0], image_size[1])  # height, width
    padding_mode = config.get('preprocessing.padding_mode', 'average')
    sequence_length = config.get('preprocessing.sequence_length', 29)
    
    # Create output directory for preprocessed data
    # output_dir = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path) + '_lip_regions')
    output_dir = "/home/aswath/Projects/capstone/multimodel_lipread/video/data_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lip region extractor
    extractor = LipRegionExtractor(target_size=target_size, padding_mode=padding_mode)
    
    # Find all video files in the dataset
    video_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files in the dataset")
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Get relative path to maintain directory structure
        rel_path = os.path.relpath(video_path, dataset_path)
        output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.npy')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract lip sequence
        try:
            lip_sequence = extractor.extract_lip_sequence(video_path, num_frames=sequence_length)
            
            # Save the lip sequence
            np.save(output_path, lip_sequence)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # Release resources
    extractor.close()
    
    print(f"Preprocessing completed. Preprocessed data saved to {output_dir}")


def main():
    """
    Main function to process GLips dataset for visual speech recognition.
    """
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/video/config/visual_config.yaml"
    process_dataset(config_path)


if __name__ == '__main__':
    main()
