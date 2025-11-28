import os
import re
import json
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from config.config import load_config
from openai import OpenAI


# Global OpenAI client
client = None


def encode_image(image_path):
    """Encode an image as base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def group_frames_by_sequence(dataset_path):
    """Group frames by (word + sequence_id) extracted from filenames."""
    files = os.listdir(dataset_path)
    sequences = defaultdict(list)
    pattern = r"(\w+)_(\d+-\d+)_frame(\d+)\.jpg"

    for file in files:
        match = re.match(pattern, file)
        if match:
            word = match.group(1)
            sequence_id = match.group(2)
            frame_num = int(match.group(3))
            key = f"{word}_{sequence_id}"
            sequences[key].append((frame_num, os.path.join(dataset_path, file)))

    # Sort frames inside each sequence
    for key in sequences:
        sequences[key].sort()

    # Return only sorted file paths
    return {key: [path for _, path in frames] for key, frames in sequences.items()}


def process_sequence(word, sequence_id, frame_paths):
    """Send the first three frames of a sequence to GPT model for analysis."""
    selected_frames = frame_paths[:3]
    encoded_frames = [encode_image(frame) for frame in selected_frames]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe the speaker’s emotional cues from their facial expressions and eye movement in these video frames of someone pronouncing '{word}'."
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                            }
                            for frame in encoded_frames
                        ],
                    ],
                }
            ],
            max_tokens=500,
        )

        return {
            "word": word,
            "sequence_id": sequence_id,
            "description": response.choices[0].message.content,
        }

    except Exception as e:
        print(f"Error processing {word}_{sequence_id}: {e}")
        return None


def main():
    global client

    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues/config/cues_config.yaml"
    config = load_config(config_path)

    cue_dataset_path = config.get("cue_dataset.input_dir")
    description_save_path = config.get("cue_dataset.output_dir")
    api_key = config.get("main.openai_api_key")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    mode = "train"
    word = "aufgaben"  # aufgaben, dagegen, lieber, sein

    dataset_path = os.path.join(cue_dataset_path, mode, word)

    sequences = group_frames_by_sequence(dataset_path)
    results = []


    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for key, frame_paths in sequences.items():
            word, sequence_id = key.split("_", 1)
            futures.append(
                executor.submit(process_sequence, word, sequence_id, frame_paths)
            )

        for future in futures:
            result = future.result()
            if result:
                results.append(result)

                # Save intermittent checkpoints
                if len(results) % 10 == 0:
                    with open(
                        f"{description_save_path}/interim_results_emotion_{word}_{mode}.json",
                        "w",
                    ) as f:
                        json.dump(results, f, indent=2)

    # Save final combined results
    with open(
        f"{description_save_path}/lipreading_analysis_results_emotion_{word}_{mode}.json",
        "w",
    ) as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
