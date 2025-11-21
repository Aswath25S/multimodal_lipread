"""
Test script for visual preprocessing using MediaPipe Face Mesh.
This script processes a single video and displays the extracted lip regions.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from visual_preprocessing import LipRegionExtractor

def test_single_video(video_path, output_dir=None):
    """
    Process a single video and display the extracted lip regions.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save the extracted lip regions
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Create lip region extractor
    extractor = LipRegionExtractor(target_size=(44, 44), padding_mode='average')
    
    # Extract lip sequence
    lip_sequence = extractor.extract_lip_sequence(video_path, num_frames=29)
    
    # Display the sequence
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    axes = axes.flatten()
    
    # Show the first 29 frames (or fewer if the video is shorter)
    for i in range(min(29, len(lip_sequence))):
        if i < len(axes):
            axes[i].imshow(cv2.cvtColor(lip_sequence[i], cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'lip_sequence.png')
        plt.savefig(output_path)
        print(f"Saved lip sequence to {output_path}")
        
        # Save the sequence as a NumPy array
        output_path = os.path.join(output_dir, 'lip_sequence.npy')
        np.save(output_path, lip_sequence)
        print(f"Saved lip sequence array to {output_path}")
    
    plt.show()
    
    # Release resources
    extractor.close()


def main():
    """
    Main function to test visual preprocessing.
    """
    video_path = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4/lipread_files/aufgaben/train/aufgaben_0001-0863.mp4"
    output_path = "/home/aswath/Projects/capstone/multimodel_lipread/video/data_utils"
    
    test_single_video(video_path, output_path)


if __name__ == '__main__':
    main()
