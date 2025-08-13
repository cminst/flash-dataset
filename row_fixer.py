#!/usr/bin/env python3
"""
Update all rows in labeling_tasks.csv where status is "assigned":
- Change status to "unassigned"
- Clear assigned_at column value
"""

import csv
import shutil
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CSV_PATH = Path("labeling_tasks.csv")          # file to edit
BACKUP_SUFFIX = ".bak"                         # backup file suffix
# ----------------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    """Read CSV into list of OrderedDict rows (preserves column order)."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")
        sys.exit(1)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows back to CSV, preserving original column order."""
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f"Failed to write {path}: {exc}")
        sys.exit(1)


def backup_file(path: Path) -> None:
    """Create a backup copy next to the original."""
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    try:
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")
    except Exception as exc:
        print(f"Could not create backup ({backup_path}): {exc}")


def main() -> None:
    if not CSV_PATH.is_file():
        print(f"File not found: {CSV_PATH}")
        sys.exit(1)

    # Load CSV
    rows = read_csv(CSV_PATH)
    if not rows:
        print("CSV appears to be empty.")
        sys.exit(1)

    # Verify required columns exist
    fieldnames = list(rows[0].keys())
    required_cols = {'status', 'assigned_at'}
    missing = required_cols - set(fieldnames)
    if missing:
        print(f"Missing required columns: {missing}")
        sys.exit(1)

    # Process all rows
    updated_count = 0
    for row in rows:
        if row['status'] == "assigned":
            row['status'] = "unassigned"
            row['assigned_at'] = ""  # Clear assigned_at value
            updated_count += 1

    if updated_count == 0:
        print("No rows with 'assigned' status found.")
        return

    # Show summary of changes
    print(f"\nUpdated {updated_count} rows:")
    print("  - Changed 'assigned' status to 'unassigned'")
    print("  - Cleared 'assigned_at' column values")

    # Save changes
    backup_file(CSV_PATH)
    write_csv(CSV_PATH, rows, fieldnames)
    print("\nAll changes saved successfully!")


if __name__ == "__main__":
    main()