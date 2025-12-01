import json
import os
import re


def sanitize_descriptions(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = []
    replaced_count = 0

    for entry in data:
        word = entry["word"]
        desc = entry["description"]

        # Regex pattern to match word safely (case-insensitive, whole word)
        # Handles: aufgaben, "aufgaben", 'aufgaben'
        pattern = re.compile(
            rf'(["\']?)\b{re.escape(word)}\b(["\']?)',
            flags=re.IGNORECASE
        )

        # Replace with placeholder
        new_desc, n = pattern.subn(r'"target word"', desc)

        if n > 0:
            replaced_count += 1

        updated.append({
            "word": entry["word"],
            "sequence_id": entry["sequence_id"],
            "description": new_desc
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"✅ Done!")
    print(f"📄 Input file:  {input_json_path}")
    print(f"📄 Output file: {output_json_path}")
    print(f"🔎 Entries modified: {replaced_count} / {len(data)}")


# -------------------------------------------------
# USAGE
# -------------------------------------------------
if __name__ == "__main__":
    base_path = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"
    input_file = f"{base_path}/Descriptions_Environment/lipreading_analysis_results_environment_sein_val.json"
    output_file = f"{base_path}/Corrected_Environment/lipreading_analysis_results_environment_sein_val.json"

    sanitize_descriptions(input_file, output_file)
