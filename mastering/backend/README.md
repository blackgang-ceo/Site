# BLACKGANG Mix Studio — One-Click Mastering (backend)

This directory contains the FastAPI backend and DSP pipeline for an AI-driven one-click mastering tool.

Structure:
- main.py — FastAPI app exposing /api/master endpoint
- dsp_pipeline.py — offline mastering pipeline that uses librosa, pyloudnorm, pedalboard (if available), scipy & ffmpeg for limiting
- requirements.txt — Python deps


