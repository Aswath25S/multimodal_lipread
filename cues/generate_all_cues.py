import os
import re
import json
import base64
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.config import load_config
from openai import OpenAI
from threading import Lock


# Global OpenAI client
client = None
rate_lock = Lock()
last_request_time = 0


def encode_image(image_path):
    """Encode an image file as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def group_frames_by_sequence(dataset_path):
    """Group frames by word + sequence ID extracted from filenames."""
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

    for key in sequences:
        sequences[key].sort()

    return {key: [path for _, path in frames] for key, frames in sequences.items()}


def rate_limit_guard(min_interval=22):
    """Ensure minimum spacing between API calls globally."""
    global last_request_time

    with rate_lock:
        now = time.time()
        elapsed = now - last_request_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            print(f"[RateGuard] Sleeping {sleep_time:.1f}s to respect limits...")
            time.sleep(sleep_time)

        last_request_time = time.time()


def process_sequence(word, sequence_id, frame_paths, emotion):
    """Send frames safely to OpenAI with retry logic + rate control."""
    selected_frames = frame_paths[:3]
    encoded_frames = [encode_image(frame) for frame in selected_frames]

    retries = 6
    
    if emotion == "emotion":
        text = f"Describe the speaker’s emotional cues from their facial expressions and eye movement in these video frames of someone pronouncing '{word}'."
    else:
        text = f"Describe the environment around the speaker, include information on light, background scene, place, etc."

    for attempt in range(retries):
        try:
            # Enforce spacing across threads
            rate_limit_guard()

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
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

            print(f"[Success] {word}_{sequence_id}")

            return {
                "word": word,
                "sequence_id": sequence_id,
                "description": response.choices[0].message.content,
            }

        except Exception as e:
            err = str(e).lower()

            # Handle rate limits gracefully
            if "rate limit" in err or "429" in err:
                wait = 25
                print(f"[429] Waiting {wait}s before retry... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue

            print(f"[Error] {word}_{sequence_id}: {e}")
            return None

    print(f"[Abort] Retries exceeded for {word}_{sequence_id}")
    return None


def main(mode, word, emotion):
    global client

    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues/config/cues_config.yaml"
    config = load_config(config_path)

    cue_dataset_path = config.get("cue_dataset.input_dir")
    description_save_path = config.get("cue_dataset.output_dir")
    api_key = config.get("main.openai_api_key")

    client = OpenAI(api_key=api_key)

    dataset_path = os.path.join(cue_dataset_path, mode, word)

    sequences = group_frames_by_sequence(dataset_path)
    results = []

    # ✅ Faster but safe: up to 2 concurrent workers
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for key, frame_paths in sequences.items():
            word, sequence_id = key.split("_", 1)
            futures.append(
                executor.submit(process_sequence, word, sequence_id, frame_paths, emotion)
            )

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

                # Save checkpoint every 10 results
                if len(results) % 10 == 0:
                    with open(
                        f"{description_save_path}/interim_results_{emotion}_{word}_{mode}.json",
                        "w",
                    ) as f:
                        json.dump(results, f, indent=2)

        print("Sequences processed - ", len(futures))

    with open(
        f"{description_save_path}/lipreading_analysis_results_{emotion}_{word}_{mode}.json",
        "w",
    ) as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    modes = ["train", "test", "val"]
    words = ["aufgaben", "dagegen", "lieber", "sein"]
    emotions = ["emotion", "environment"]

    for mode in modes:
        for word in words:
            for emotion in emotions:
                print("\n\n\n")
                print("Mode: ", mode)
                print("Word: ", word)
                print("Emotion: ", emotion)
                print("\n\n\n")
                main(mode, word, emotion)
            print('------------------------------------------------------------------------')
            print("\n\n\n")