import matplotlib.pyplot as plt
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import yt_dlp

# Function to plot dataset statistics
def plot_dataset_stats(dataset):
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