import pandas as pd
from fastapi import FastAPI, HTTPException
import os
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import threading
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "downloaded_clips"

DB_FILE = 'labeling_tasks.csv'
TASK_TIMEOUT_MINUTES = 15

lock = threading.Lock()

app = FastAPI()


class Label(BaseModel):
    row_id: int
    revised_caption: str
    build_up: float
    peak_start: float
    peak_end: float
    drop_off: float


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main labeling interface."""
    with open("index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/video/{filename:path}")
async def get_video(filename: str):
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid video path.")

    file_path = VIDEO_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

    return FileResponse(file_path, media_type="video/mp4")


@app.get("/get-next-task")
def get_next_task():
    """Assigns a video to a user.
    Prioritizes stale tasks.
    Handles cases where a user refreshes or closes the page."""
    with lock:
        df = pd.read_csv(DB_FILE, parse_dates=['assigned_at'])
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(minutes=TASK_TIMEOUT_MINUTES)

        # Look for stale tasks first
        stale_tasks = df[(df['status'] == 'assigned')
                         & (df['assigned_at'] < timeout_threshold)]

        task_to_assign = None
        if not stale_tasks.empty:
            task_index = stale_tasks.index[0]
            print(f"Re-assigning stale task with row_id: {df.loc[task_index, 'row_id']}")
        else:
            # If no stale tasks, get a new unassigned one
            unassigned_tasks = df[df['status'] == 'unassigned']
            if unassigned_tasks.empty:
                raise HTTPException(status_code=404,
                                    detail="All tasks are completed or assigned!")
            task_index = unassigned_tasks.index[0]

        # Assign the task
        df.loc[task_index, 'status'] = 'assigned'
        df.loc[task_index, 'assigned_at'] = now
        df.to_csv(DB_FILE, index=False)

        task = df.loc[task_index].where(pd.notnull(df.loc[task_index]), None).to_dict()
        return task


@app.post("/submit-labels")
def submit_labels(label: Label):
    """Receives and saves the completed labels for a video."""
    with lock:
        df = pd.read_csv(DB_FILE)
        mask = df['row_id'] == label.row_id
        if not mask.any():
            raise HTTPException(status_code=404,
                                detail=f"Row ID {label.row_id} not found.")
        
        idx = df.index[mask][0]

        df.loc[idx, 'revised_caption'] = label.revised_caption
        df.loc[idx, 'build_up'] = label.build_up
        df.loc[idx, 'peak_start'] = label.peak_start
        df.loc[idx, 'peak_end'] = label.peak_end
        df.loc[idx, 'drop_off'] = label.drop_off
        df.loc[idx, 'status'] = 'completed'
        df.loc[idx, 'assigned_at'] = None

        df.to_csv(DB_FILE, index=False)
        return {"status": "success", "row_id": label.row_id}


@app.post("/delete-task/{row_id}")
def delete_task(row_id: int):
    """Marks a task as 'bad_quality' so it's removed from the queue."""
    with lock:
        df = pd.read_csv(DB_FILE)
        mask = df['row_id'] == row_id
        if not mask.any():
            raise HTTPException(status_code=404,
                                detail=f"Row ID {row_id} not found.")
        
        idx = df.index[mask][0]
        df.loc[idx, 'status'] = 'bad_quality'
        df.loc[idx, 'assigned_at'] = None
        df.to_csv(DB_FILE, index=False)

        return {"status": "deleted", "row_id": row_id}


@app.get("/get-progress")
def get_progress():
    """Calculates and returns the overall labeling progress."""
    df = pd.read_csv(DB_FILE)
    total = len(df[df['status'] != 'bad_quality'])
    completed = len(df[df['status'] == 'completed'])
    return {"completed": completed, "total": total}