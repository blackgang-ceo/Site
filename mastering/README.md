# BLACKGANG Mastering

This folder contains a one-click mastering backend and a minimal Next.js frontend page.

Quick dev notes:

1. Backend (FastAPI)
   - cd mastering/backend
   - python -m venv .venv
   - source .venv/bin/activate
   - pip install -r requirements.txt
   - uvicorn main:app --reload --port 8000

2. Frontend (Next.js)
   - cd mastering/frontend
   - npm install
   - npm run dev

The frontend expects the backend to be available at /api/master; configure a proxy in your Next.js dev server (or call the backend directly at http://localhost:8000/master).
