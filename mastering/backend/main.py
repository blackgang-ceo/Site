"""
main.py
FastAPI server exposing a /api/master endpoint that accepts audio files, runs the mastering pipeline and returns the results.
"""
import os
import shutil
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional

from dsp_pipeline import master_file

app = FastAPI(title="BLACKGANG One-Click Mastering API")

# CORS for local dev / Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(tempfile.gettempdir()) / "blackgang_mastering"
WORK_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}


@app.post("/api/master")
async def api_master(file: UploadFile = File(...), target_lufs: Optional[float] = -10.0):
    """
    Accept an uploaded audio file and run the mastering pipeline synchronously.
    Returns JSON with download URLs (files) or direct file responses.

    For production you would enqueue background tasks and return job ids.
    """
    # basic validation
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aiff', '.aac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    job_id = str(uuid.uuid4())

    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / file.filename
    # stream-upload to disk to avoid loading huge files into memory
    with open(input_path, 'wb') as f:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk:
                break
            f.write(chunk)

    # produce two masters (streaming and high-res)
    streaming_out = job_dir / (input_path.stem + "_streaming_16bit_44100.wav")
    highres_out = job_dir / (input_path.stem + "_highres_24bit_48000.wav")

    try:
        # run mastering pipeline (synchronous). master_file will raise on error.
        master_file(str(input_path), str(streaming_out), target_lufs=float(target_lufs), out_samplerate=44100, bitdepth=16)
        master_file(str(input_path), str(highres_out), target_lufs=float(target_lufs), out_samplerate=48000, bitdepth=24)
    except Exception as e:
        # cleanup
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Mastering failed: {e}")

    # For simplicity return direct download links to two files
    return JSONResponse({
        "job_id": job_id,
        "streaming_master": f"/api/download/{job_id}/{streaming_out.name}",
        "highres_master": f"/api/download/{job_id}/{highres_out.name}",
    })


@app.get('/api/download/{job_id}/{filename}')
async def download(job_id: str, filename: str):
    job_dir = WORK_DIR / job_id
    candidate = job_dir / filename
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(candidate), filename=filename, media_type='audio/wav')


@app.get('/api/status/{job_id}')
async def status(job_id: str):
    job_dir = WORK_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    files = [p.name for p in job_dir.glob('*')]
    return {"job_id": job_id, "files": files}
