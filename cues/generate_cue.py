import os
import re
import json
import base64
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import openai
# Define the path to your dataset
dataset_path = "Screenshots/extracted_frames/train/aufgaben"
openai.api_key = "test"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Group frames by video sequence
def group_frames_by_sequence():
    files = os.listdir(dataset_path)
    sequences = defaultdict(list)
    
    pattern = r'(\w+)_(\d+-\d+)_frame(\d+)\.jpg'
    
    for file in files:
        match = re.match(pattern, file)
        if match:
            word = match.group(1)
            sequence_id = match.group(2)
            frame_num = int(match.group(3))
            key = f"{word}_{sequence_id}"
            sequences[key].append((frame_num, os.path.join(dataset_path, file)))
    
    for key in sequences:
        sequences[key].sort()
    
    return {key: [path for _, path in frames] for key, frames in sequences.items()}

def process_sequence(word, sequence_id, frame_paths):
    # Use only the first 3 frames to stay within token limit
    selected_frames = frame_paths[:3]

    encoded_frames = [encode_image(frame) for frame in selected_frames]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",  # Use GPT-4 Vision if available
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe the speaker’s emotional cues from their facial expressions and eye movement in these video frames of someone pronouncing '{word}'."
                            # "text": f"Describe the environment around the speaker, include information on light, background scene, place, etc."
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame}"
                                }
                            }
                            for frame in encoded_frames
                        ]
                    ]
                }
            ],
            max_tokens=500  # Limit the response size
        )

        return {
            "word": word,
            "sequence_id": sequence_id,
            "description": response["choices"][0]["message"]["content"]
        }
    except Exception as e:
        print(f"Error processing {word}_{sequence_id}: {e}")
        return None

def main():
    sequences = group_frames_by_sequence()
    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for key, frame_paths in sequences.items():
            word, sequence_id = key.split('_', 1)
            futures.append(executor.submit(process_sequence, word, sequence_id, frame_paths))

        for future in futures:
            result = future.result()
            if result:
                results.append(result)
                if len(results) % 10 == 0:
                    with open("interim_results_emotion_aufgaben.json", "w") as f:
                        json.dump(results, f, indent=2)

    with open("Descriptions/lipreading_analysis_results_emotion_aufgaben.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()