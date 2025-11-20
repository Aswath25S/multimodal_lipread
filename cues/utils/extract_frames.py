import cv2
import os

def extract_frames(video_path, output_dir, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Skipping empty or unreadable video: {video_path}")
        return

    frame_ids = [int(i * total_frames / num_frames) for i in range(num_frames)]
    basename = os.path.splitext(os.path.basename(video_path))[0]

    for idx, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"{basename}_frame{idx+1}.jpg")
            cv2.imwrite(out_path, frame)

    cap.release()

# -------- CONFIGURATION --------
input_dir = os.path.join("lipread_files")  # Top-level folder with word subfolders
output_root = os.path.join("Screenshots", "extracted_frames", "train")

os.makedirs(output_root, exist_ok=True)
# -------- PROCESSING --------
for root, _, files in os.walk(input_dir):
    # Skip folders that aren't 'train'
    if os.path.basename(root).lower() != "train":
        continue
    for file in files:
        if file.endswith(".mp4"):
            word = os.path.basename(os.path.dirname(root))  # e.g., "aufgaben"
            word_dir = os.path.join(output_root, word)
            os.makedirs(word_dir, exist_ok=True)

            video_path = os.path.join(root, file)
            print(f"Extracting from: {video_path}")
            extract_frames(video_path, word_dir)
