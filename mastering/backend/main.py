# Mastering backend FastAPI server
# main.py — exposes /process endpoint that accepts an audio file and returns processed masters (streaming)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import tempfile
from pathlib import Path
from dsp_pipeline import MasteringEngine
import asyncio

app = FastAPI(title="BLACKGANG One-Click Mastering - Backend",
              description="FastAPI backend that runs an automated mastering chain.")

# configure CORS so the Next.js frontend (localhost:3000) can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a single engine instance; it is stateless per call but we keep it for convenience
engine = MasteringEngine()

OUTPUT_DIR = Path(tempfile.gettempdir()) / "blackgang_mastering_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/process")
async def process_audio(file: UploadFile = File(...), target_lufs: float = -10.0):
    """
    Accepts an uploaded audio file (wav/mp3/m4a) and returns JSON with links to
    the processed files (streaming master and hi-res master). Files are stored
    temporarily in the system temp directory and will be cleaned up by caller or periodic job.

    Query params:
    - target_lufs: desired integrated LUFS (default -10.0). We'll clamp to [-14, -8].
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # clamp target lufs to a safe commercial range
    target_lufs = max(-14.0, min(-8.0, float(target_lufs)))

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".wav", ".mp3", ".m4a", ".aiff", ".flac"]:
        # let ffmpeg handle some conversions but validate extension roughly
        pass

    # create safe temp files
    input_fd, input_path = tempfile.mkstemp(suffix=suffix, prefix="bgang_input_")
    os.close(input_fd)

    try:
        # stream uploaded file to disk (avoids loading huge files into memory)
        with open(input_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # process - this is run in threadpool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        # produce two outputs: streaming (16/44.1) and hires (24/48)
        out_streaming = OUTPUT_DIR / (Path(file.filename).stem + "_streaming.wav")
        out_hires = OUTPUT_DIR / (Path(file.filename).stem + "_hires.wav")

        def run_pipeline():
            engine.process_to_targets(
                input_path=str(input_path),
                out_streaming=str(out_streaming),
                out_hires=str(out_hires),
                target_lufs=target_lufs,
                tp_ceiling_db=-1.0,
            )
            return True

        await loop.run_in_executor(None, run_pipeline)

        # Return URLs (for local dev, direct file responses)
        return JSONResponse({
            "streaming_master": f"/download/{out_streaming.name}",
            "hires_master": f"/download/{out_hires.name}",
            "analysis": engine.get_last_report(),
        })

    finally:
        # Remove the uploaded temp file — pipeline created its own derived files
        try:
            os.remove(input_path)
        except Exception:
            pass


@app.get("/download/{name}")
async def download(name: str):
    # Security: prevent path traversal
    safe = OUTPUT_DIR / Path(name).name
    if not safe.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(safe), media_type="audio/wav", filename=safe.name)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
