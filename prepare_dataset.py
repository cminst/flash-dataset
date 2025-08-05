# prepare_dataset.py
import pandas as pd
from datasets import load_dataset
import os

DB_FILE = 'labeling_tasks.csv'
VIDEO_DIR = 'video'

def run():
    print("Checking if database file already exists...")
    if os.path.exists(DB_FILE):
        print(f"'{DB_FILE}' already exists. Skipping preparation.")
        print("If you want to start over, please delete the file and run this script again.")
        return

    print("Fetching dataset from Hugging Face Hub: qingy2024/FLASH-Unlabelled...")
    # Load your specific dataset and split
    ds = load_dataset("qingy2024/FLASH-Unlabelled", split="train")
    df = ds.to_pandas()

    print(f"Loaded {len(df)} rows.")

    # --- Add columns for labeling state and new labels ---
    # Status: unassigned, assigned, completed, bad_quality
    df['status'] = 'unassigned'
    df['assigned_to'] = None  # Could be used for user tracking later
    df['assigned_at'] = pd.NaT # For handling stale/abandoned tasks
    
    # Your desired labels, initialized
    df['build_up'] = -1.0
    df['peak_start'] = -1.0
    df['peak_end'] = -1.0
    df['drop_off'] = -1.0
    
    # Ensure video paths are correct for our structure
    # The HF dataset already has the "video/" prefix, which is perfect.
    print("Verifying video paths...")
    if not all(df['video'].str.startswith(f'{VIDEO_DIR}/')):
        print(f"Warning: Some video paths do not seem to start with '{VIDEO_DIR}/'.")
        print("Example path:", df['video'].iloc[0])

    # Save to CSV, which will be our "database"
    df.to_csv(DB_FILE, index=False)
    print(f"Successfully created '{DB_FILE}' with {len(df)} tasks.")
    print("You can now start the main server with 'uvicorn main:app --reload'")

if __name__ == "__main__":
    run()
