import cv2
import os
from config.config import load_config

def extract_frames(video_path, output_dir, num_frames=3):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the video can't be read or is empty, skip it
    if total_frames == 0:
        print(f"Skipping empty or unreadable video: {video_path}")
        return

    # Select evenly spaced frame indices (e.g., 3 frames across the video)
    frame_ids = [int(i * total_frames / num_frames) for i in range(num_frames)]
    # Extract the base filename without extension
    basename = os.path.splitext(os.path.basename(video_path))[0]

    # Loop through selected frames
    for idx, frame_id in enumerate(frame_ids):
        # Set video position to the target frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        # If successfully read, save frame as an image
        if ret:
            out_path = os.path.join(output_dir, f"{basename}_frame{idx+1}.jpg")
            cv2.imwrite(out_path, frame)

    # Release video file handle
    cap.release()

if __name__ == "__main__":
    # Load configuration file
    config_path = "./config/cues_config.yaml"
    config = load_config(config_path)

    # Get directories from config
    input_dir = config.get("dataset.input_dir")
    mode = "test"  # Only process validation videos
    
    # Create root output directory for extracted frames
    output_root = os.path.join(config.get("dataset.output_dir"), mode)
    os.makedirs(output_root, exist_ok=True)

    # -------- PROCESSING --------
    for root, _, files in os.walk(input_dir):
        # Process only folders matching the mode (e.g., "val")
        if os.path.basename(root).lower() != mode:
            continue

        for file in files:
            # Only process .mp4 video files
            if file.endswith(".mp4"):
                # Determine the word/category based on parent folder
                word = os.path.basename(os.path.dirname(root))  # e.g., "aufgaben"
                
                # Create output directory for this word if missing
                word_dir = os.path.join(output_root, word)
                os.makedirs(word_dir, exist_ok=True)

                # Build full path to the video file
                video_path = os.path.join(root, file)
                print(f"Extracting from: {video_path}")

                # Extract frames and save them
                extract_frames(video_path, word_dir)
