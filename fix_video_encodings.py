#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.style import Style
    from rich.text import Text
except ImportError:
    print("Error: This script requires the 'rich' library.")
    print("Install it using: pip install rich")
    sys.exit(1)

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

def get_video_files(input_folder: Path) -> List[Path]:
    """Get all .mp4 files (case-insensitive) in input folder"""
    video_files = []
    extensions = [".mp4", ".MP4"]
    
    for entry in input_folder.iterdir():
        if entry.is_file() and entry.suffix in extensions:
            video_files.append(entry)
    
    if not video_files:
        console.print(f"[yellow]No MP4 files found in {input_folder}[/yellow]")
        sys.exit(0)
    
    return video_files

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

def process_video(input_path: Path, output_path: Path) -> bool:
    """Process single video with ffmpeg"""
    global PROCESSED, ERRORS
    
    try:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-c:v", "libopenh264", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-y",
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown ffmpeg error"
            ERRORS.append((input_path.name, error_msg))
            return False
        return True
    except Exception as e:
        ERRORS.append((input_path.name, str(e)))
        return False
    finally:
        PROCESSED += 1

def update_display(progress, stats_task, video_name: str = "-"):
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
        (video_name, "italic magenta"),
    )
    stats_task.update(stats_text)

def main():
    # Verify ffmpeg installation
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
    input_folder = get_folder("Input video folder", "video")
    output_folder = get_folder("Output folder", "fixed")
    
    # Get video files
    videos = get_video_files(input_folder)
    total = len(videos)
    
    # Prepare progress display
    progress, stats = create_progress_display(total)
    task_id = progress.add_task("Processing videos", total=total)
    
    # Process videos with live display
    with Live(
        Group(
            Panel(stats, title="Video Processing", border_style="blue", box=box.ROUNDED),
            progress
        ),
        console=console,
        refresh_per_second=10
    ) as live:
        for video in videos:
            output_path = output_folder / video.name
            
            # Update current file
            update_display(progress, live, video.name)
            
            # Process video
            success = process_video(video, output_path)
            
            # Update progress bar
            if success:
                progress.update(task_id, advance=1, refresh=True)
            else:
                progress.update(task_id, advance=1, refresh=True)
        
        # Final update
        update_display(progress, live)
        progress.update(task_id, completed=total, refresh=True)
    
    # Show summary
    console.print("\n[bold]Processing Complete:[/bold]")
    console.print(f" • Total Videos: [cyan]{total}[/cyan]")
    console.print(f" • Successfully Processed: [green]{total - len(ERRORS)}[/green]")
    
    if ERRORS:
        console.print(f" • Failed: [red]{len(ERRORS)}[/red]\n")
        
        error_table = Table(
            title="Failed Videos", 
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        error_table.add_column("File", style="cyan", width=30)
        error_table.add_column("Error", style="red", width=50)
        
        for filename, error in ERRORS[:5]:  # Show first 5 errors
            error_table.add_row(filename, error[:70] + "..." if len(error) > 70 else error)
        
        if len(ERRORS) > 5:
            console.print(error_table)
            console.print(f"[yellow]... and {len(ERRORS) - 5} more errors (check log for details)[/yellow]")
        else:
            console.print(error_table)
        
        # Save full error log
        log_path = output_folder / "processing_errors.log"
        with open(log_path, "w") as f:
            for filename, error in ERRORS:
                f.write(f"=== {filename} ===\n{error}\n\n")
        
        console.print(f"\n[italic]Full error log saved to: {log_path}[/italic]")
    else:
        console.print("\n[green]All videos processed successfully! 🎉[/green]")
    
    console.print(f"\n[bold]Output saved to:[/bold] {output_folder}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Processing interrupted by user[/red]")
        sys.exit(1)
