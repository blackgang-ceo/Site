# BLACKGANG Mix Studio — Mastering backend (FastAPI)
# main.py
# Production-ready FastAPI app that exposes a single /master endpoint.
# Accepts multipart file upload (wav/mp3/m4a), runs the DSP pipeline, and returns a processed WAV/MP3.

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import tempfile
import asyncio
from pathlib import Path

from dsp_pipeline import MasteringEngine

app = FastAPI(title="BLACKGANG Mastering Engine")

# Allow the frontend (likely running on localhost:3000) to talk to the API during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared engine instance (lightweight). It spawns per-call workers.
engine = MasteringEngine()

@app.post('/master')
async def master_audio(
    file: UploadFile = File(...),
    target: str = Form('streaming'),
    target_lufs: float = Form(-10.0),
    format: str = Form('wav')
):
    """
    Accepts an uploaded audio file, runs mastering, and returns the mastered file.
    - target: 'streaming' or 'highres' (controls bitdepth/sample-rate in output)
    - target_lufs: desired integrated LUFS (e.g. -11 to -9 recommended)
    - format: 'wav' or 'mp3'

    The endpoint processes the file synchronously and streams the output back.
    """
    # Validate extension quickly
    filename = Path(file.filename).name
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    if ext not in ('wav', 'mp3', 'm4a', 'aiff', 'flac'):
        raise HTTPException(status_code=400, detail='Unsupported audio format')

    # Save upload to a secure temp file (auto-clean on close)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as in_tmp:
            shutil.copyfileobj(file.file, in_tmp)
            in_path = Path(in_tmp.name)

        # Choose output params
        if target == 'highres':
            out_sr = 48000
            out_bits = 24
        else:
            out_sr = 44100
            out_bits = 16

        # Create temp output path
        out_suffix = f'.{format}' if format != 'wav' else '.wav'
        out_path = Path(tempfile.gettempdir()) / (f'mastered-{uuid.uuid4().hex}{out_suffix}')

        # Run the CPU-bound pipeline in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            engine.process_file,
            str(in_path),
            str(out_path),
            target_lufs,
            out_sr,
            out_bits,
            format
        )

        if not result:
            raise HTTPException(status_code=500, detail='Processing failed')

        # Return the file as a response and schedule cleanup
        response = FileResponse(path=str(out_path), filename=f'mastered{out_suffix}', media_type='application/octet-stream')

        # Cleanup uploaded input file
        try:
            os.remove(in_path)
        except Exception:
            pass

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health():
    return JSONResponse({'status':'ok'})
