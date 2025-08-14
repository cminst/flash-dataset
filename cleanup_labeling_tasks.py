#!/usr/bin/env python3
"""
Cleanup script for labeling tasks CSV data.
Processes peak data to remove overlaps, fix timing issues, and prepare for dataset creation.
"""

import csv
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import re

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class Peak:
    """Represents a single peak with timing data"""
    build_up: float
    peak_start: float
    peak_end: float
    drop_off: float
    caption: str
    video_id: str
    row_id: str
    true_peak_start: float = None
    true_peak_end: float = None

    def __post_init__(self):
        # Ensure caption ends with period
        if not self.caption.endswith('.'):
            self.caption += '.'

class LabelingTaskCleaner:
    """Main class for cleaning labeling tasks data"""

    def __init__(self, input_csv: str, output_csv: str = None, huggingface_dataset: str = None, reference_dataset: str = "qingy2024/FLASH-Unlabelled"):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv) if output_csv else self.input_csv.parent / f"{self.input_csv.stem}_cleaned.csv"
        self.huggingface_dataset = huggingface_dataset
        self.reference_dataset = reference_dataset

        # Load reference dataset for true time calculation
        self.timing_data = self._load_timing_data()

        # Statistics tracking
        self.stats = {
            'total_rows': 0,
            'rows_with_peaks': 0,
            'total_peaks': 0,
            'removed_overlaps': 0,
            'removed_invalid_timing': 0,
            'fixed_drop_off': 0,
            'added_periods': 0,
            'final_peaks': 0
        }

    def _load_timing_data(self) -> Dict[str, Dict[str, float]]:
        """Load timing data from reference dataset"""
        try:
            from datasets import load_dataset

            console.print(f"[blue]Loading reference dataset: {self.reference_dataset}[/blue]")
            dataset = load_dataset(self.reference_dataset)

            timing_data = {}
            for row in dataset['train']:
                # Extract video_id and row_id from video path
                video_path = row['video']
                filename = Path(video_path).name
                if filename.endswith('.mp4'):
                    filename = filename[:-4]

                parts = filename.split('_')
                if len(parts) >= 2:
                    video_id = '_'.join(parts[:-1])
                    row_id = parts[-1]
                else:
                    video_id = filename
                    row_id = "0"

                key = f"{video_id}_{row_id}"
                timing_data[key] = {
                    'start_time': float(row['start_time']),
                    'end_time': float(row['end_time'])
                }

            console.print(f"[green]Loaded timing data for {len(timing_data)} videos[/green]")
            return timing_data

        except ImportError:
            console.print("[yellow]HuggingFace datasets library not found. Using relative timing.[/yellow]")
            console.print("[yellow]Install with: pip install datasets[/yellow]")
            return {}
        except Exception as e:
            console.print(f"[yellow]Error loading reference dataset: {e}[/yellow]")
            console.print("[yellow]Using relative timing instead.[/yellow]")
            return {}

    def extract_video_info(self, video_path: str) -> Tuple[str, str, float, float]:
        """Extract video ID, row ID and original timing from video path"""
        # Pattern is downloaded_clips/{video_id}_{row_id}.mp4
        filename = Path(video_path).name
        if filename.endswith('.mp4'):
            filename = filename[:-4]

        parts = filename.split('_')
        if len(parts) >= 2:
            video_id = '_'.join(parts[:-1])
            row_id = parts[-1]
        else:
            video_id = filename
            row_id = "0"

        # Get original timing from reference dataset
        key = f"{video_id}_{row_id}"
        if key in self.timing_data:
            original_start = self.timing_data[key]['start_time']
            original_end = self.timing_data[key]['end_time']
        else:
            original_start = 0.0
            original_end = 14.0

        return video_id, row_id, original_start, original_end

    def calculate_true_time(self, peak: Peak, original_start: float, original_end: float) -> Peak:
        """Calculate true time in original video for peak start and end"""
        # The clip is cropped from (original_start - 7) to (original_end + 7)
        # Clip duration is (original_end + 7) - (original_start - 7) = original_end - original_start + 14
        # Peak timing is relative to the clip start (which is original_start - 7)

        clip_start = original_start - 7.0
        peak.true_peak_start = clip_start + peak.peak_start
        peak.true_peak_end = clip_start + peak.peak_end

        return peak

    def parse_peaks_from_string(self, peaks_str: str, video_path: str, row_id: str) -> List[Peak]:
        """Parse peaks from JSON string in CSV"""
        if not peaks_str or peaks_str == '[]':
            return []

        try:
            peaks_data = json.loads(peaks_str.replace('""', '"'))
            video_id, _, original_start, original_end = self.extract_video_info(video_path)

            peaks = []
            for peak_data in peaks_data:
                peak = Peak(
                    build_up=float(peak_data['build_up']),
                    peak_start=float(peak_data['peak_start']),
                    peak_end=float(peak_data['peak_end']),
                    drop_off=float(peak_data['drop_off']),
                    caption=peak_data['caption'],
                    video_id=video_id,
                    row_id=row_id
                )
                # Calculate true time using original video timing
                self.calculate_true_time(peak, original_start, original_end)
                peaks.append(peak)

            return peaks
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse peaks for {video_path}: {e}")
            return []

    def check_overlap(self, peak1: Peak, peak2: Peak) -> bool:
        """Check if two peaks overlap in peak start and end with same caption"""
        # Check if captions are the same (case insensitive, ignoring trailing periods)
        caption1 = peak1.caption.rstrip('.').lower().strip()
        caption2 = peak2.caption.rstrip('.').lower().strip()

        if caption1 != caption2:
            return False

        # Check for overlap in peak start and end
        return not (peak1.peak_end <= peak2.peak_start or peak2.peak_end <= peak1.peak_start)

    def remove_overlapping_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """Remove overlapping peaks with same caption, keeping the first one"""
        if len(peaks) <= 1:
            return peaks

        # Sort peaks by peak_start
        peaks.sort(key=lambda p: p.peak_start)

        kept_peaks = []
        removed_count = 0

        for i, current_peak in enumerate(peaks):
            should_remove = False

            # Check against all kept peaks
            for kept_peak in kept_peaks:
                if self.check_overlap(current_peak, kept_peak):
                    should_remove = True
                    removed_count += 1
                    break

            if not should_remove:
                kept_peaks.append(current_peak)

        self.stats['removed_overlaps'] += removed_count
        return kept_peaks

    def validate_peak_timing(self, peak: Peak) -> bool:
        """Validate peak timing and fix issues"""
        is_valid = True

        # Check if build_up is after peak_start
        if peak.build_up > peak.peak_start:
            is_valid = False
            self.stats['removed_invalid_timing'] += 1
            return False

        # Check if drop_off is before peak_end
        if peak.drop_off < peak.peak_end:
            peak.drop_off = peak.peak_end
            self.stats['fixed_drop_off'] += 1

        return is_valid

    def process_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """Process peaks through all cleanup steps"""
        if not peaks:
            return []

        # Step 1: Remove overlapping peaks
        peaks = self.remove_overlapping_peaks(peaks)

        # Step 2: Validate and fix timing
        valid_peaks = []
        for peak in peaks:
            if self.validate_peak_timing(peak):
                valid_peaks.append(peak)

        return valid_peaks

    def create_huggingface_dataset(self, cleaned_data: List[Dict[str, Any]]) -> None:
        """Create and push HuggingFace dataset if requested"""
        if not self.huggingface_dataset:
            return

        try:
            from datasets import Dataset
            import pandas as pd

            # Remove true_peak_start and true_peak_end from peaks data and minor tweaks for caption strings.
            for row in cleaned_data:
                if 'peaks' in row:
                    peaks_data = json.loads(row['peaks'])
                    for peak in peaks_data:
                        del peak['true_peak_start']
                        del peak['true_peak_end']

                        # Replace \u00a0 or continuous \u00a0s with space in caption
                        peak['caption'] = re.sub(r'\u00a0+', ' ', peak['caption'])

                        # For captions with "pinata"
                        peak['caption'] = re.sub(r'\u00f1', 'n', peak['caption'])

                    row['peaks'] = json.dumps(peaks_data)

            # Convert to DataFrame
            df = pd.DataFrame(cleaned_data)

            # Create dataset
            dataset = Dataset.from_pandas(df)

            # Cleanup dataset
            dataset = dataset.remove_columns([
                'status',
                'assigned_to',
                'assigned_at',
                'build_up',
                'peak_start',
                'peak_end',
                'drop_off',
                'caption',
                'revised_caption'
            ])

            dataset = dataset.shuffle(seed=42)

            # Push to hub
            dataset.push_to_hub(self.huggingface_dataset, private=True)

            console.print(f"[green]Dataset pushed to: https://huggingface.co/datasets/{self.huggingface_dataset}[/green]")

        except ImportError:
            console.print("[yellow]HuggingFace datasets library not found. Skipping dataset creation.[/yellow]")
            console.print("[yellow]Install with: pip install datasets[/yellow]")
        except Exception as e:
            console.print(f"[red]Error creating HuggingFace dataset: {e}[/red]")

    def process_csv(self) -> List[Dict[str, Any]]:
        """Main processing function"""
        console.print(f"[bold]Processing:[/bold] {self.input_csv}")
        console.print(f"[bold]Output:[/bold] {self.output_csv}")

        cleaned_data = []

        with open(self.input_csv, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames

            for row in Progress().track(
                list(reader),
                description="Processing rows"
            ):
                self.stats['total_rows'] += 1

                # Parse peaks
                peaks = self.parse_peaks_from_string(
                    row.get('peaks', '[]'),
                    row.get('video', ''),
                    row.get('row_id', '0')
                )

                if peaks:
                    self.stats['rows_with_peaks'] += 1
                    self.stats['total_peaks'] += len(peaks)

                # Process peaks
                processed_peaks = self.process_peaks(peaks)

                if processed_peaks:
                    self.stats['final_peaks'] += len(processed_peaks)

                    # Convert peaks back to JSON string
                    peaks_json = json.dumps([
                        {
                            'build_up': peak.build_up,
                            'peak_start': peak.peak_start,
                            'peak_end': peak.peak_end,
                            'drop_off': peak.drop_off,
                            'caption': peak.caption,
                            'true_peak_start': peak.true_peak_start,
                            'true_peak_end': peak.true_peak_end
                        }
                        for peak in processed_peaks
                    ])

                    # Update row
                    cleaned_row = row.copy()
                    cleaned_row['peaks'] = peaks_json

                    # Count added periods
                    original_peaks = peaks
                    for original, processed in zip(original_peaks, processed_peaks):
                        if original.caption != processed.caption:
                            self.stats['added_periods'] += 1

                    cleaned_data.append(cleaned_row)

        # Write cleaned data
        if cleaned_data:
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cleaned_data)

        return cleaned_data

    def print_summary(self):
        """Print processing summary"""
        console.print("\n[bold blue]Cleanup Summary[/bold blue]")
        console.print(Panel(
            f"""Total rows processed: {self.stats['total_rows']}
Rows with peaks: {self.stats['rows_with_peaks']}
Original peaks: {self.stats['total_peaks']}
Removed overlapping peaks: {self.stats['removed_overlaps']}
Removed invalid timing peaks: {self.stats['removed_invalid_timing']}
Fixed drop-off timing: {self.stats['fixed_drop_off']}
Added periods to captions: {self.stats['added_periods']}
Final peaks: {self.stats['final_peaks']}""",
            title="Statistics",
            border_style="blue"
        ))

        if self.stats['total_peaks'] > 0:
            reduction_rate = (self.stats['total_peaks'] - self.stats['final_peaks']) / self.stats['total_peaks'] * 100
            console.print(f"Peak reduction rate: {reduction_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Clean up labeling tasks CSV data")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: input_cleaned.csv)")
    parser.add_argument("--huggingface", help="HuggingFace dataset name to push to (e.g., 'username/dataset-name')")
    parser.add_argument("--reference-dataset", default="qingy2024/FLASH-Unlabelled",
                       help="Reference dataset for timing calculation (default: qingy2024/FLASH-Unlabelled)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize cleaner
    cleaner = LabelingTaskCleaner(
        input_csv=args.input_csv,
        output_csv=args.output,
        huggingface_dataset=args.huggingface,
        reference_dataset=args.reference_dataset
    )

    try:
        # Process data
        cleaned_data = cleaner.process_csv()

        # Create HuggingFace dataset if requested
        if args.huggingface and cleaned_data:
            cleaner.create_huggingface_dataset(cleaned_data)

        # Print summary
        cleaner.print_summary()

        console.print(f"\n[green]Cleanup complete! Output saved to: {cleaner.output_csv}[/green]")

    except Exception as e:
        console.print(f"[red]Error during processing: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
