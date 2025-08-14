#!/usr/bin/env python3
"""
find_and_fix_av1.py

Scan a directory for videos, report which are broken (e.g., AV1 "Missing Sequence Header"),
and try to fix them by remuxing. If remux fails, optionally transcode to H.264.

Usage:
  python find_and_fix_av1.py /path/to/videos [--transcode-on-fail]

Creates: /path/to/videos/fixed_clips
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Optional OpenCV check for a quick decode probe
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.padding import Padding
from rich.panel import Panel

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".ts", ".m4s"}

console = Console()

def run(cmd: list, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def have_tool(name: str) -> bool:
    return shutil.which(name) is not None

def ffprobe_stream(path: Path) -> Optional[dict]:
    """
    Return primary video stream and format info via ffprobe, or None on failure.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=index,codec_name,codec_type,avg_frame_rate,r_frame_rate,extradata_size,profile,pix_fmt,"
        "color_space,color_transfer,color_primaries",
        "-show_entries", "format=format_name,format_long_name,nb_streams,probe_score",
        "-of", "json",
        str(path),
    ]
    try:
        proc = run(cmd)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        if not streams:
            return None
        return {"stream": streams[0], "format": fmt}
    except Exception:
        return None

def quick_decode_ok(path: Path) -> Tuple[bool, float]:
    """
    Try to open and grab one frame with OpenCV. Also read FPS.
    Returns (ok, fps). OpenCV is optional.
    """
    if not HAS_CV2:
        return (True, 0.0)  # skip test if cv2 is missing
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return (False, 0.0)
    ok, _ = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return (bool(ok), float(fps))

def looks_like_segment(path: Path) -> bool:
    # Heuristic for DASH/HLS fragments that will not have headers
    name = path.name.lower()
    return path.suffix.lower() == ".m4s" or "chunk" in name or "seg" in name or "init" in name

def is_suspect_missing_header(meta: Optional[dict]) -> bool:
    """
    Judge likely missing AV1 sequence header from ffprobe fields.
    """
    if not meta:
        return True
    s = meta["stream"]
    codec = s.get("codec_name", "")
    extradata_size = s.get("extradata_size", None)
    # Classic symptom: AV1 with zero or absent extradata
    if codec == "av1":
        if extradata_size in (0, None):
            return True
    return False

def decide_output_path(in_path: Path, out_dir: Path) -> Path:
    # Keep same name and extension when possible. Convert .ts or .m4s to .mp4 for better tooling.
    ext = in_path.suffix.lower()
    if ext in {".ts", ".m4s"}:
        out_name = in_path.stem + ".mp4"
    else:
        out_name = in_path.name
    out_path = out_dir / out_name
    # Avoid accidental overwrite
    i = 1
    while out_path.exists():
        out_path = out_dir / f"{in_path.stem}_fixed_{i}{out_path.suffix}"
        i += 1
    return out_path

def try_remux(in_path: Path, out_path: Path) -> bool:
    # For MP4 and MOV, add faststart. For others, just copy.
    ext = in_path.suffix.lower()
    if ext in {".mp4", ".mov", ".m4v"} or out_path.suffix.lower() == ".mp4":
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts",
            "-i", str(in_path),
            "-map", "0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-map", "0",
            "-c", "copy",
            str(out_path),
        ]
    proc = run(cmd)
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0

def try_transcode_h264(in_path: Path, out_path: Path) -> bool:
    # Safe fallback: video to H.264, audio copied. Works even if AV1 support is spotty in your stack.
    # Note: this decodes the source, so it still fails if the file is truly corrupt or starts midstream.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-map", "0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "22",
        "-c:a", "copy",
        str(out_path.with_suffix(".mp4")),
    ]
    proc = run(cmd)
    out_h264 = out_path.with_suffix(".mp4")
    return proc.returncode == 0 and out_h264.exists() and out_h264.stat().st_size > 0

def process_file(path: Path, out_dir: Path, transcode_on_fail: bool) -> dict:
    info = {
        "file": str(path),
        "status": "ok",
        "detail": "",
        "fixed_path": "",
    }

    if looks_like_segment(path):
        info["status"] = "skip"
        info["detail"] = "Likely a DASH/HLS segment. Needs manifest or init segment."
        return info

    meta = ffprobe_stream(path)
    suspect = is_suspect_missing_header(meta)
    decode_ok, fps = quick_decode_ok(path)

    bad = False
    reason = []
    if suspect:
        bad = True
        reason.append("AV1 missing header suspected")
    if not decode_ok or fps <= 0:
        bad = True
        reason.append("decode or FPS probe failed")

    if not bad:
        info["status"] = "good"
        info["detail"] = f"codec={meta['stream'].get('codec_name','?')}, fps={fps:.3f}" if meta else f"fps={fps:.3f}"
        return info

    # Try to fix
    out_path = decide_output_path(path, out_dir)
    remux_ok = try_remux(path, out_path)
    if remux_ok:
        # Verify
        meta2 = ffprobe_stream(out_path)
        suspect2 = is_suspect_missing_header(meta2)
        decode_ok2, fps2 = quick_decode_ok(out_path)
        if not suspect2 and decode_ok2 and fps2 > 0:
            info["status"] = "fixed"
            info["fixed_path"] = str(out_path)
            info["detail"] = f"remuxed: fps={fps2:.3f}, codec={meta2['stream'].get('codec_name','?') if meta2 else '?'}"
            return info

    if transcode_on_fail:
        out_path2 = out_path  # base name
        trans_ok = try_transcode_h264(path, out_path2)
        if trans_ok:
            final = out_path2.with_suffix(".mp4")
            decode_ok3, fps3 = quick_decode_ok(final)
            info["status"] = "fixed"
            info["fixed_path"] = str(final)
            info["detail"] = f"transcoded to H.264, fps={fps3:.3f}" if decode_ok3 else "transcoded to H.264"
            return info

    info["status"] = "failed"
    joined_reason = "; ".join(reason) if reason else "unknown"
    info["detail"] = f"attempted remux{' and transcode' if transcode_on_fail else ''}; still not decodable ({joined_reason})"
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing video files")
    parser.add_argument("--transcode-on-fail", action="store_true", help="If remux does not fix, try H.264 transcode")
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        console.print(f"[red]Folder not found or not a directory:[/red] {root}")
        sys.exit(1)

    if not have_tool("ffprobe") or not have_tool("ffmpeg"):
        console.print("[red]ffmpeg/ffprobe not found on PATH. Please install ffmpeg first.[/red]")
        sys.exit(1)

    out_dir = root / "fixed_clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    files.sort()

    header = Panel.fit(
        f"[bold]Scanning {len(files)} files in[/bold] [cyan]{root}[/cyan]\n"
        f"Output: [green]{out_dir}[/green]\n"
        f"OpenCV check: {'on' if HAS_CV2 else 'off'} | Transcode on fail: {args.transcode_on_fail}",
        border_style="blue",
    )
    console.print(header)

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as prog:
        task = prog.add_task("Processing videos...", total=len(files))
        for p in files:
            res = process_file(p, out_dir, args.transcode_on_fail)
            results.append(res)
            prog.advance(task)

    # Summaries
    counts = {"good": 0, "fixed": 0, "failed": 0, "skip": 0, "ok": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    table = Table(title="Results", title_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    table.add_column("Output", overflow="fold")

    for r in results:
        status_color = {
            "good": "green",
            "fixed": "green",
            "failed": "red",
            "skip": "yellow",
            "ok": "green",
        }.get(r["status"], "white")
        table.add_row(
            Path(r["file"]).name,
            f"[{status_color}]{r['status']}[/{status_color}]",
            r["detail"],
            r["fixed_path"],
        )

    console.print(table)

    summary = (
        f"[green]good={counts.get('good',0)}[/green], "
        f"[green]fixed={counts.get('fixed',0)}[/green], "
        f"[yellow]skip={counts.get('skip',0)}[/yellow], "
        f"[red]failed={counts.get('failed',0)}[/red]"
    )
    console.print(Padding(f"Summary: {summary}", (1, 0, 0, 0)))

    if counts.get("skip", 0) > 0:
        console.print(
            "[yellow]Tip:[/yellow] Skipped items look like DASH or HLS fragments. "
            "You usually need the manifest (MPD or M3U8) or the init segment to reconstruct a playable file."
        )

if __name__ == "__main__":
    main()
