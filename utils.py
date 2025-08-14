import matplotlib.pyplot as plt
import numpy as np
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
def plot_peak_dataset_stats(dataset, thumos_duration_cap: int = 40):
    """
    Plot various statistics and distributions from a peak dataset.

    Parameters
    ----------
    dataset : list
        A list of dictionaries, each containing a 'peaks' field that holds action segment data.
    thumos_duration_cap : int, optional
        Maximum duration to consider when comparing with THUMOS'14 dataset, default is 40.

    Returns
    -------
    None
        This function displays plots but does not return any values.

    Notes
    -----
    * Creates a 2x2 grid of plots showing caption length distribution, action duration distribution,
      a comparison histogram with THUMOS'14 data, and a boxplot comparison.
    * Expects each row in the dataset to have a 'peaks' field containing a list of dictionaries,
      with each dictionary having 'caption', 'peak_start', and 'peak_end' fields.
    * Will attempt to load THUMOS'14 data from a directory relative to the script location.
    """

    # Extract caption lengths and action durations from the peaks
    caption_lengths = []
    durations = []
    for row in dataset:
        peaks = eval(row['peaks'])
        for peak in peaks:
            caption_lengths.append(len(peak['caption'].split()))

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
                    if len(parts) >= 3:
                        start_time = float(parts[-2])
                        end_time = float(parts[-1])

                        thumos_durations.append(end_time - start_time)

    _, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Caption Word Lengths (Top-Left)
    ax[0,0].hist(caption_lengths, bins=20, color='skyblue', edgecolor='black')
    ax[0,0].set_title('Caption Word Length Distribution (from Peaks)')
    ax[0,0].set_xlabel('Number of Words')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_xlim(left=0)
    ax[0,0].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Action Durations Histogram (Top-Right)
    ax[0,1].hist(durations, bins=20, color='salmon', edgecolor='black')
    ax[0,1].set_title('Action Duration Distribution (from Peaks)')
    ax[0,1].set_xlabel('Duration (seconds)')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_xlim(left=0)
    ax[0,1].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 3: Duration Comparison Histogram (Bottom-Left)
    if durations and thumos_durations:
        min_duration = min(min(durations), min(thumos_durations))
        max_duration = max(max(durations), max(thumos_durations))
        bins = 30

        ax[1,0].hist(durations, bins=bins, range=(min_duration, max_duration),
                     alpha=0.7, color='red', label='Our Dataset', density=True)
        ax[1,0].hist(thumos_durations, bins=bins, range=(min_duration, max_duration),
                     alpha=0.7, color='purple', label='THUMOS\'14', density=True)
        ax[1,0].set_title('Action Duration Distribution Comparison')
        ax[1,0].set_xlabel('Duration (seconds)')
        ax[1,0].set_ylabel('Percentage of Videos')
        ax[1,0].set_xlim(left=0, right=thumos_duration_cap)
        ax[1,0].grid(axis='y', alpha=0.3, linestyle='--')
        ax[1,0].legend()

    # Plot 4: Duration Comparison Boxplot (Bottom-Right)
    if durations and thumos_durations:
        labels = ['FLASH', "THUMOS'14 (< cap)"]
        ax[1,1].boxplot([durations, thumos_durations],
                        labels=labels,
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        medianprops=dict(color='red', linewidth=2))
        ax[1,1].set_title('Action Duration Comparison (Boxplot)')
        ax[1,1].set_ylabel('Duration (seconds)')
        ax[1,1].set_ylim(bottom=-0.5, top=20)
        ax[1,1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    # Stats table
    if durations:
        def summarize(arr, name):
            arr_np = np.asarray(arr, dtype=float)
            stats = {
                'dataset': name,
                'count': int(arr_np.size),
                'mean': float(np.mean(arr_np)) if arr_np.size else float('nan'),
                'median': float(np.median(arr_np)) if arr_np.size else float('nan'),
                'q25': float(np.percentile(arr_np, 25)) if arr_np.size else float('nan'),
                'q75': float(np.percentile(arr_np, 75)) if arr_np.size else float('nan'),
                'min': float(np.min(arr_np)) if arr_np.size else float('nan'),
                'max': float(np.max(arr_np)) if arr_np.size else float('nan'),
            }
            return stats

        rows = []
        rows.append(summarize(durations, 'FLASH'))
        if thumos_durations:
            rows.append(summarize(thumos_durations, "THUMOS'14"))

        headers = ['dataset', 'count', 'mean', 'median', 'q25', 'q75', 'min', 'max']

        try:
            from tabulate import tabulate
            print()
            print(tabulate(
                [[r[h] for h in headers] for r in rows],
                headers=headers,
                tablefmt='fancy_grid',
                floatfmt='.2f',
            ))
        except Exception:
            col_widths = {h: max(len(h), max(len(f"{r[h]:.2f}") if isinstance(r[h], float) else len(str(r[h])) for r in rows)) for h in headers}
            def fmt_cell(v):
                return f"{v:.2f}" if isinstance(v, float) else str(v)
            header_line = ' | '.join(h.ljust(col_widths[h]) for h in headers)
            sep_line = '-+-'.join('-' * col_widths[h] for h in headers)
            print(header_line)
            print(sep_line)
            for r in rows:
                print(' | '.join(fmt_cell(r[h]).ljust(col_widths[h]) for h in headers))

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
