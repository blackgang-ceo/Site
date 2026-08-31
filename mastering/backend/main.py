from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import shutil
import tempfile
import asyncio
from typing import Dict

from .dsp_pipeline import process_mastering

app = FastAPI(title="BLACKGANG Mastering Engine")

# Allow local frontend dev to talk to backend — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKDIR = os.path.join(os.getcwd(), "mastering", "jobs")
os.makedirs(WORKDIR, exist_ok=True)

# In-memory job status store (simple). For production, replace with Redis or DB.
jobs: Dict[str, Dict] = {}


@app.post('/api/master')
async def upload_and_master(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Accept a single file, store to temp file, start background processing
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.aiff', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input")
    # Save upload to disk in streaming fashion
    with open(input_path, 'wb') as f:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk:
                break
            f.write(chunk)

    jobs[job_id] = {
        'status': 'queued',
        'phase': 'queued',
        'progress': 0,
        'result': None,
        'job_dir': job_dir
    }

    # Launch background processing
    background_tasks.add_task(_run_processing, job_id, input_path)

    return JSONResponse({ 'job_id': job_id })


async def _run_processing(job_id: str, input_path: str):
    jobs[job_id]['status'] = 'running'
    try:
        def progress_cb(phase, pct):
            jobs[job_id]['phase'] = phase
            jobs[job_id]['progress'] = pct

        # Run the heavy DSP in a threadpool so it doesn't block event loop
        loop = asyncio.get_event_loop()
        out_files = await loop.run_in_executor(None, process_mastering, input_path, jobs[job_id]['job_dir'], progress_cb)

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['result'] = out_files
        jobs[job_id]['phase'] = 'complete'
        jobs[job_id]['progress'] = 100
    except Exception as exc:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['phase'] = 'error'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['error'] = str(exc)


@app.get('/api/status/{job_id}')
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail='Job not found')
    data = jobs[job_id].copy()
    # Do not leak internal paths
    data.pop('job_dir', None)
    return JSONResponse(data)


@app.get('/api/download/{job_id}/{variant}')
async def download(job_id: str, variant: str):
    # variant = streaming | highres
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail='Job not found')
    res = jobs[job_id].get('result')
    if not res:
        raise HTTPException(status_code=404, detail='Result not ready')
    if variant not in res:
        raise HTTPException(status_code=404, detail='Variant not found')
    path = res[variant]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='File missing')
    return FileResponse(path, filename=os.path.basename(path), media_type='application/octet-stream')
