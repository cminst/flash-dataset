import pandas as pd
import json
from pathlib import Path

DB_FILE = Path('labeling_tasks.csv')
BACKUP_FILE = Path('labeling_tasks.backup.csv')

def to_peaks_row(row):
    # If peaks already valid JSON, keep as-is
    peaks_val = row.get('peaks', None)
    if isinstance(peaks_val, str) and peaks_val.strip():
        try:
            json.loads(peaks_val)
            return peaks_val
        except Exception:
            pass  # fall through to rebuild

    def as_num(x):
        try:
            return float(x)
        except Exception:
            return None

    bu = as_num(row.get('build_up', None))
    ps = as_num(row.get('peak_start', None))
    pe = as_num(row.get('peak_end', None))
    do = as_num(row.get('drop_off', None))

    has_legacy = (
        bu is not None and bu >= 0 and
        ps is not None and ps >= 0 and
        pe is not None and pe >= 0 and
        do is not None and do >= 0
    )

    if has_legacy:
        cap = row.get('revised_caption', '') or ''
        peaks = [{
            'build_up': float(bu),
            'peak_start': float(ps),
            'peak_end': float(pe),
            'drop_off': float(do),
            'caption': cap,
        }]
    else:
        peaks = []

    return json.dumps(peaks)


def main():
    if not DB_FILE.exists():
        raise SystemExit(f"Could not find {DB_FILE}")

    print(f"Backing up {DB_FILE} -> {BACKUP_FILE}")
    BACKUP_FILE.write_bytes(DB_FILE.read_bytes())

    print(f"Loading {DB_FILE}")
    df = pd.read_csv(DB_FILE)

    # Ensure peaks column exists
    if 'peaks' not in df.columns:
        df['peaks'] = None

    print("Converting legacy single-peak fields to peaks JSON where needed...")
    df['peaks'] = df.apply(to_peaks_row, axis=1)

    # Drop legacy columns: start_time, end_time, action_duration
    legacy_cols = [c for c in ['start_time', 'end_time', 'action_duration'] if c in df.columns]
    if legacy_cols:
        print(f"Dropping legacy columns: {', '.join(legacy_cols)}")
        df = df.drop(columns=legacy_cols)

    df.to_csv(DB_FILE, index=False)
    print("Migration complete.")
    print("Summary:")
    try:
        # Basic summary of how many rows have non-empty peaks
        peaks_counts = df['peaks'].apply(lambda s: len(json.loads(s)) if isinstance(s, str) and s.strip() else 0)
        print(f"  Rows with >=1 peak: {(peaks_counts > 0).sum()} / {len(df)}")
    except Exception:
        pass

if __name__ == '__main__':
    main()

