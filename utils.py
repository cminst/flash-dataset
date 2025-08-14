import matplotlib.pyplot as plt
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import yt_dlp

# Function to plot caption length and action duration distributions
def plot_caption_and_duration_distributions(dataset):
    # Calculate caption lengths and action durations
    caption_lengths = [len(row['revised_caption'].split()) for row in dataset]
    durations = [row['action_duration'] for row in dataset]

    # Create subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Caption Word Lengths
    ax1.hist(caption_lengths, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Caption Word Length Distribution')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Action Durations
    ax2.hist(durations, bins=20, color='salmon', edgecolor='black')
    ax2.set_title('Action Duration Distribution')
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(left=0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Final layout
    plt.tight_layout()
    plt.show()

# Function to plot caption length and action duration distributions from peaks data
def plot_peak_dataset_stats(dataset):
    # Extract caption lengths and action durations from the 'peaks' field
    caption_lengths = []
    durations = []
    for row in dataset:
        peaks = eval(row['peaks'])  # Convert string representation of list to actual list
        for peak in peaks:
            # Caption length in words
            caption_lengths.append(len(peak['caption'].split()))
            # Action duration computed as peak_end - peak_start
            durations.append(peak['peak_end'] - peak['peak_start'])

    # Load THUMOS'14 data
    thumos_durations = []
    thumos_dir = Path(__file__).parent / "THUMOS14"
    if thumos_dir.exists():
        for txt_file in thumos_dir.glob("*_val.txt"):
            # Skip the ambiguous file
            if "Ambiguous" in txt_file.name:
                continue

            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:  # Ensure we have the required data
                        start_time = float(parts[-2])
                        end_time = float(parts[-1])
                        duration = end_time - start_time
                        if duration <= 40:
                            thumos_durations.append(duration)

    # Create subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

    # Plot 1: Caption Word Lengths
    ax1.hist(caption_lengths, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Caption Word Length Distribution (from Peaks)')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Action Durations (Peak Duration)
    ax2.hist(durations, bins=20, color='salmon', edgecolor='black')
    ax2.set_title('Action Duration Distribution (from Peaks)')
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(left=0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 3: Action Duration Distribution (Percentage Comparison)
    if durations and thumos_durations:
        min_duration = min(min(durations), min(thumos_durations))
        max_duration = max(max(durations), max(thumos_durations))
        bins = 30

        # Plot histograms as percentages
        ax3.hist(durations, bins=bins, range=(min_duration, max_duration),
                 alpha=0.7, color='red', label='Our Dataset', density=True)
        ax3.hist(thumos_durations, bins=bins, range=(min_duration, max_duration),
                 alpha=0.7, color='purple', label='THUMOS\'14', density=True)
        ax3.set_title('Action Duration Distribution Comparison')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Percentage of Videos')
        ax3.set_xlim(left=0)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.legend()

    # Final layout
    plt.tight_layout()
    plt.show()

# Function to download a YouTube clip
def download_youtube_clip(
    video_id: str,
    start_sec: float,
    end_sec: float,
    output_file: Union[str, Path],
) -> Path:
    """
    Download a YouTube segment as an MP4 video-only clip.

    Parameters
    ----------
    video_id : str
        The YouTube video ID (the part after `v=` in the URL).
    start_sec : float
        Desired segment start time in seconds.
    end_sec : float
        Desired segment end time in seconds.
    output_file : str or Path
        Where to save the resulting MP4.

    Returns
    -------
    Path
        Path to the saved MP4.

    Notes
    -----
    * The function grabs an extra 7-second margin on both sides,
      then trims with FFmpeg.
    * Requires `yt-dlp>=2024.04` and a working `ffmpeg` binary.
    """

    # Ensure sane input
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    # Calculate clip start and end times
    clip_start = max(0, start_sec - 7)
    clip_end = end_sec + 7
    clip_duration = clip_end - clip_start

    # Construct the YouTube URL
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Set the output file path
    output_file = Path(output_file).with_suffix(".mp4")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set the temporary file path
        tmp_path = Path(tmpdir) / "source.mp4"

        # Download the best MP4 video-only stream
        ydl_opts = {
            "format": "bestvideo[ext=mp4]",  # video only, always MP4
            "outtmpl": str(tmp_path),
            "quiet": True,
            "cookiefile": "ytc.txt"
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Trim and strip audio
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                        # overwrite if exists
            "-ss", str(clip_start),      # seek to start
            "-i", str(tmp_path),
            "-t", str(clip_duration),    # keep only needed duration
            "-an",                       # drop audio
            "-c", "copy",                # avoid re-encoding
            str(output_file),
        ]
        subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

    return output_file
