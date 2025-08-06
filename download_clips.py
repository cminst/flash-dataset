#!/usr/bin/env python3
import os
import sys
import json
import yt_dlp
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from utils import download_youtube_clip
from rich.text import Text

console = Console()
ERRORS = []
PROCESSED = 0

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
    # Example: 'video/v_Ffi7vDa3C2I.mp4' -> 'Ffi7vDa3C2I'
    filename = Path(video_path).name
    if filename.endswith('.mp4'):
        base = filename[:-4]
    else:
        base = filename
    
    # Remove 'v_' prefix if present
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

def create_progress_display(total: int) -> Tuple[Progress, Text]:
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
    
    stats = Text.assemble(
        ("Processed: ", "bold"),
        ("0", "cyan"),
        (" / ", "bold"),
        (str(total), "cyan"),
        (" • Failed: ", "bold"),
        ("0", "red"),
        ("\n"),
        ("Current: ", "bold"),
        ("-", "italic"),
    )
    
    return progress, stats

def process_task(task: Dict, output_dir: Path) -> bool:
    """Process single video download task"""
    global PROCESSED, ERRORS
    output_path = output_dir / task['output_filename']
    
    try:
        download_youtube_clip(
            video_id=task['video_id'],
            start_sec=task['start_sec'],
            end_sec=task['end_sec'],
            output_file=output_path
        )
        return True
    except Exception as e:
        ERRORS.append((task['output_filename'], str(e)))
        return False
    finally:
        PROCESSED += 1

def update_display(progress, stats_task, task_name: str = "-"):
    """Update progress display components"""
    stats_text = Text.assemble(
        ("Processed: ", "bold"),
        (str(PROCESSED), "cyan"),
        (" / ", "bold"),
        (str(progress.tasks[0].total), "cyan"),
        (" • Failed: ", "bold"),
        (str(len(ERRORS)), "red"),
        ("\n"),
        ("Current: ", "bold"),
        (task_name, "italic magenta"),
    )
    stats_task.update(stats_text)

def main():
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
    
    # Prepare progress display
    progress, stats = create_progress_display(total)
    task_id = progress.add_task("Downloading clips", total=total)
    
    # Process clips with live display
    with Live(
        Group(
            Panel(stats, title="Download Progress", border_style="blue", box=box.ROUNDED),
            progress
        ),
        console=console,
        refresh_per_second=10
    ) as live:
        for task in tasks:
            # Update current file
            update_display(progress, live, task['output_filename'])
            
            # Process clip
            success = process_task(task, output_dir)
            
            # Update progress bar
            progress.update(task_id, advance=1, refresh=True)
        
        # Final update
        update_display(progress, live)
        progress.update(task_id, completed=total, refresh=True)
    
    # Show summary
    console.print("\n[bold]Download Complete:[/bold]")
    console.print(f" • Total Clips: [cyan]{total}[/cyan]")
    console.print(f" • Successfully Downloaded: [green]{total - len(ERRORS)}[/green]")
    
    if ERRORS:
        console.print(f" • Failed: [red]{len(ERRORS)}[/red]\n")
        
        error_table = Table(
            title="Failed Downloads", 
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        error_table.add_column("File", style="cyan", width=40)
        error_table.add_column("Error", style="red", width=60)
        
        for filename, error in ERRORS[:5]:
            error_table.add_row(
                filename, 
                error[:70] + "..." if len(error) > 70 else error
            )
        
        if len(ERRORS) > 5:
            console.print(error_table)
            console.print(f"[yellow]... and {len(ERRORS) - 5} more errors (check log for details)[/yellow]")
        else:
            console.print(error_table)
        
        # Save full error log
        log_path = output_dir / "download_errors.log"
        with open(log_path, "w") as f:
            for filename, error in ERRORS:
                f.write(f"=== {filename} ===\n{error}\n\n")
        
        console.print(f"\n[italic]Full error log saved to: {log_path}[/italic]")
    else:
        console.print("\n[green]All clips downloaded successfully! 🎉[/green]")
    
    console.print(f"\n[bold]Downloaded clips saved to:[/bold] {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Download interrupted by user[/red]")
        sys.exit(1)