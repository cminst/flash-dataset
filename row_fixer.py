#!/usr/bin/env python3
"""
Replace the caption inside the JSON stored in the `peaks` column of
row 1017 of labeling_tasks.csv.

Usage:
    python replace_caption.py   # run from the directory that contains labeling_tasks.csv
"""

import csv
import json
import shutil
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CSV_PATH = Path("labeling_tasks.csv")          # file to edit
ROW_NUMBER = 1017                              # 1‑based row number as you gave it
TARGET_CAPTION = "The moment when the man lands in the sand."
BACKUP_SUFFIX = ".bak"                         # backup file suffix
# ----------------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    """Read the CSV into a list of OrderedDict rows (preserves column order)."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"❌ Failed to read {path}: {exc}")
        sys.exit(1)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows back to CSV, preserving the original column order."""
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f"❌ Failed to write {path}: {exc}")
        sys.exit(1)


def backup_file(path: Path) -> None:
    """Create a simple backup copy next to the original."""
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    try:
        shutil.copy2(path, backup_path)
        print(f"🔐 Backup created: {backup_path}")
    except Exception as exc:
        print(f"⚠️ Could not create backup ({backup_path}): {exc}")
        # Continue anyway – the user can decide whether to abort.


def main() -> None:
    if not CSV_PATH.is_file():
        print(f"❌ File not found: {CSV_PATH}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    rows = read_csv(CSV_PATH)
    if not rows:
        print("❌ CSV appears to be empty.")
        sys.exit(1)

    # Determine zero‑based index (CSV rows are 1‑based in the description)
    idx = ROW_NUMBER - 1
    if idx < 0 or idx >= len(rows):
        print(f"❌ Row number {ROW_NUMBER} is out of range (file has {len(rows)} rows).")
        sys.exit(1)

    row = rows[idx]
    fieldnames = list(row.keys())  # preserve original column order

    # ------------------------------------------------------------------
    # Show the row to the user
    # ------------------------------------------------------------------
    print("\n=== Row {} (original CSV line {}) ===".format(ROW_NUMBER, idx + 2))  # +2 = header + 1‑based
    for col, val in row.items():
        print(f"{col}: {val}")

    # ------------------------------------------------------------------
    # Extract and display the current caption
    # ------------------------------------------------------------------
    peaks_raw = row.get("peaks")
    if peaks_raw is None:
        print("❌ No column named 'peaks' found.")
        sys.exit(1)

    try:
        peaks_data = json.loads(peaks_raw)
    except json.JSONDecodeError as exc:
        print(f"❌ Could not parse JSON in 'peaks' column: {exc}")
        sys.exit(1)

    if not isinstance(peaks_data, list) or not peaks_data:
        print("❌ Unexpected structure: 'peaks' should be a non‑empty list.")
        sys.exit(1)

    # Show the caption(s) that exist now
    print("\nCurrent caption(s) in the JSON:")
    for i, entry in enumerate(peaks_data):
        if isinstance(entry, dict):
            caption = entry.get("caption", "<missing>")
            print(f"  [{i}] {caption}")
        else:
            print(f"  [{i}] <non‑dict entry>")

    # ------------------------------------------------------------------
    # Ask for confirmation
    # ------------------------------------------------------------------
    answer = input(
        f"\nReplace **all** captions with \"{TARGET_CAPTION}\"? (y/N): "
    ).strip().lower()
    if answer != "y":
        print("🚫 Operation cancelled by user.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Perform the replacement
    # ------------------------------------------------------------------
    for entry in peaks_data:
        if isinstance(entry, dict) and "caption" in entry:
            entry["caption"] = TARGET_CAPTION

    # Dump back to a compact JSON string that mimics the original style
    # (original example used a space after commas and a colon+space)
    new_peaks_str = json.dumps(peaks_data, ensure_ascii=False, separators=(", ", ": "))

    # Update the row in memory
    row["peaks"] = new_peaks_str

    # ------------------------------------------------------------------
    # Write changes back to CSV (with a backup)
    # ------------------------------------------------------------------
    backup_file(CSV_PATH)
    write_csv(CSV_PATH, rows, fieldnames)

    print("\n✅ Update complete! The new 'peaks' value for row {} is:".format(ROW_NUMBER))
    print(new_peaks_str)


if __name__ == "__main__":
    main()
