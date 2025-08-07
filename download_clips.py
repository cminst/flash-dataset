import os
import sys
import json
import yt_dlp
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()

def get_folder(prompt: str, default: str) -> Path:
    """Get folder path from user with validation"""
    while True:
        path = Prompt.ask(
            f"[blue]{prompt}[/blue]", 
            default=default,
            console=console
        )
        path_obj = Path(path).expanduser().resolve()
        
        if path_obj.exists() and path_obj.is_dir():
            return path_obj
        
        if path == default and not path_obj.exists():
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return path_obj
            except Exception as e:
                console.print(f"[red]Error creating default folder: {e}[/red]")
                continue
        
        console.print(f"[yellow]Directory does not exist: {path}[/yellow]")
        create = Prompt.ask("Create this directory?", choices=["y", "n"], default="y", console=console)
        if create == "y":
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return path_obj
            except Exception as e:
                console.print(f"[red]Failed to create directory: {e}[/red]")
        else:
            continue

def extract_video_id(video_path: str) -> str:
    """Extract video ID from 'video' field path"""
    filename = Path(video_path).name
    if filename.endswith('.mp4'):
        base = filename[:-4]
    else:
        base = filename

    if base.startswith('v_'):
        return base[2:]
    return base

def load_tasks(jsonl_path: str) -> List[Dict]:
    """Load tasks from JSONL file with video ID extraction"""
    tasks = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                video_id = extract_video_id(item['video'])
                output_filename = f"{video_id}_{item['row_id']}.mp4"
                tasks.append({
                    'video_id': video_id,
                    'start_sec': item['start_time'],
                    'end_sec': item['end_time'],
                    'row_id': item['row_id'],
                    'output_filename': output_filename
                })
        return tasks
    except FileNotFoundError:
        console.print(f"[red]Error: JSONL file not found at {jsonl_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading tasks: {str(e)}[/red]")
        sys.exit(1)

def create_progress_display(total: int) -> Progress:
    """Create progress tracking components"""
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    return progress

def process_task(task: Dict, output_dir: Path, resume: bool) -> bool:
    """Process a single video download task"""
    output_path = output_dir / task['output_filename']

    if resume and output_path.exists():
        console.print(f"[dim]Skipping existing file: {task['output_filename']}[/dim]")
        return True  # already downloaded

    try:
        from utils import download_youtube_clip
        download_youtube_clip(
            video_id=task['video_id'],
            start_sec=task['start_sec'],
            end_sec=task['end_sec'],
            output_file=output_path
        )
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "private" in error_msg:
            console.print(f"[yellow]Skipping private video: {task['output_filename']}[/yellow]")
            return False  # Mark as failed but continue
        elif "not a bot" in error_msg:
            console.print(f"[red bold]Stopping download due to bot detection error:[/red bold] {str(e)}")
            raise  # Stop processing
        else:
            console.print(f"[red]Error processing {task['output_filename']}: {str(e)}[/red]")
            return False  # Other error, also skip for now

def main():
    parser = argparse.ArgumentParser(description="YouTube Clip Downloader")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded clips")
    args = parser.parse_args()

    # Verify dependencies
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Error: ffmpeg not found or not working[/red]")
        console.print("Please install ffmpeg with libopenh264 support")
        sys.exit(1)

    # Get user input
    input_file = "flash_dataset_unlabelled_9.2k.jsonl"
    output_dir = get_folder("Output directory for clips", "downloaded_clips")

    # Load tasks
    tasks = load_tasks(input_file)
    total = len(tasks)
    
    if total == 0:
        console.print("[yellow]No tasks found in dataset[/yellow]")
        sys.exit(0)

    # Progress display
    progress = create_progress_display(total)
    task_id = progress.add_task("Downloading clips", total=total)

    # Live UI
    stats_text = Text.assemble(
        ("Processed: ", "bold"),
        ("0", "cyan"),
        (" / ", "bold"),
        (str(total), "cyan"),
        (" • Failed: 0", "bold red"),
        ("\nCurrent: ", "bold"),
        ("-", "italic"),
    )

    with Live(
        Group(
            Panel(stats_text, title="Download Progress", border_style="blue", box=box.ROUNDED),
            progress
        ),
        console=console,
        refresh_per_second=10
    ) as live:

        completed = 0
        failed = 0

        try:
            for task in tasks:
                # Update live display with current clip
                current_status = Text.assemble(
                    ("Processed: ", "bold"),
                    (str(completed), "cyan"),
                    (" / ", "bold"),
                    (str(total), "cyan"),
                    (" • Failed: ", "bold"),
                    (str(failed), "red"),
                    ("\nCurrent: ", "bold"),
                    (task['output_filename'], "italic magenta"),
                )
                live.update(Group(
                    Panel(current_status, title="Download Progress", border_style="blue", box=box.ROUNDED),
                    progress
                ))

                # Process the task
                if process_task(task, output_dir, args.resume):
                    completed += 1
                else:
                    failed += 1
                progress.update(task_id, advance=1, refresh=True)

        except Exception as e:
            # Only stop on specific error ("not a bot")
            error_msg = str(e).lower()
            if "not a bot" in error_msg:
                progress.update(task_id, refresh=True)
                console.print(f"\n[red bold]Stopping download due to bot detection error:[/red bold] {str(e)}")
                sys.exit(1)
            else:
                # For other exceptions, treat like task failure and continue
                failed += 1
                progress.update(task_id, advance=1, refresh=True)
                console.print(f"[yellow]Continuing after unexpected error: {str(e)}[/yellow]")

        # Final status update
        final_status = Text.assemble(
            ("Processed: ", "bold"),
            (str(completed), "cyan"),
            (" / ", "bold"),
            (str(total), "cyan"),
            (" • Failed: ", "bold"),
            (str(failed), "red"),
            ("\nCurrent: Done", "bold green"),
        )
        live.update(Group(
            Panel(final_status, title="Download Progress", border_style="blue", box=box.ROUNDED),
            progress
        ))
        progress.update(task_id, completed=total, refresh=True)

    console.print("\n[bold]Download Complete:[/bold]")
    console.print(f" • Total Clips: [cyan]{total}[/cyan]")
    console.print(f" • Successfully Downloaded: [green]{completed}[/green]")
    console.print(f" • Failed: [red]{failed}[/red]")

    if failed == 0:
        console.print("\n[green]All clips downloaded successfully! 🎉[/green]")
    else:
        console.print("\n[red]Some downloads failed. Exiting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Downloaded clips saved to:[/bold] {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Download interrupted by user[/red]")
        sys.exit(1)