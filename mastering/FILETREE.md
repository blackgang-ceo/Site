# High-level files tree for the mastering feature

This file documents the files added for the one-click mastering feature.

mastering/
├─ backend/
│  ├─ main.py                # FastAPI service: upload endpoint + job status + download
│  └─ dsp_pipeline.py        # DSP chain using pedalboard, pyloudnorm, matchering, librosa
├─ frontend/
│  └─ app/master/page.tsx    # Next.js page: drag-drop, progress ring, Wavesurfer before/after
├─ requirements.txt          # Python dependencies for the backend
├─ package.json              # Frontend dependencies (Next.js, Tailwind, Wavesurfer, Framer Motion)

README: Use the FastAPI app (mastering/backend/main.py) to run the mastering worker. The frontend page posts audio to /api/master and polls for status.
