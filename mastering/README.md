# One-click mastering feature

This commit adds a new feature branch with a backend FastAPI mastering pipeline and a minimal Next.js frontend page.

Paths:
- mastering/backend/main.py
- mastering/backend/dsp_pipeline.py
- mastering/backend/requirements.txt
- mastering/frontend/page.tsx
- mastering/frontend/package.json

Notes:
- The backend expects ffmpeg installed on the host path for limiting steps.
- Install Python deps in a virtualenv and run `uvicorn main:app --reload` from mastering/backend
- Start the frontend with Next.js dev server in mastering/frontend
