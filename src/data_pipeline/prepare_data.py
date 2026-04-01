import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(BASE_DIR, "data", "transcripts.txt")


def create_dataset(audio_dir, text_file):
    data = []

    if not os.path.exists(text_file):
        print(f"❌ Error: Text file not found at {text_file}")
        return

    with open(text_file, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} lines in transcript file.")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split by pipe; strip whitespace to prevent path errors
        parts = line.split("|")

        if len(parts) >= 2:
            file_name = parts[0].strip()
            text = parts[1].strip()

            # Logic to handle if file_name already includes .wav
            if not file_name.lower().endswith(".wav"):
                audio_path = os.path.join(audio_dir, file_name + ".wav")
            else:
                audio_path = os.path.join(audio_dir, file_name)

            if os.path.exists(audio_path):
                data.append({
                    "audio": audio_path,
                    "text": text
                })
            else:
                # DEBUG: This will tell you exactly why you have 0 matches
                print(f"⚠️ Warning: Missing audio file: {audio_path}")

    df = pd.DataFrame(data)
    
    # Save dataset.csv to the project root
    project_root = os.path.dirname(os.path.dirname(audio_dir))
    output_path = os.path.join(project_root, "dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset created: {len(df)} rows saved to {output_path}")

if __name__ == "__main__":
    # Dynamically find the project root from this script's location
    # Assumes file is at: project_root/src/data_pipeline/prepare_data.py
    current_script = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script)))
    
    target_audio_dir = os.path.join(base_dir, "data", "raw_audio")
    target_text_file = os.path.join(base_dir, "data", "transcripts.txt")
    
    create_dataset(target_audio_dir, target_text_file)
