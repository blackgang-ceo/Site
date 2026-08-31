#!/usr/bin/env bash
# tools/make_execution_bundle.sh
# Prepare a Zip execution bundle for the BLACKGANG mastering feature.
# Usage: ./tools/make_execution_bundle.sh
# Produces: blackgang_mastering_execution_bundle.zip in the current directory

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
OUT_ZIP="$ROOT_DIR/blackgang_mastering_execution_bundle.zip"
WORKDIR=$(mktemp -d)
BUNDLE_DIR="$WORKDIR/blackgang_mastering_bundle"
mkdir -p "$BUNDLE_DIR"

echo "Creating bundle in $BUNDLE_DIR"

# Files and directories to include (if they exist)
INCLUDE=(
  "mastering"
  "docker-compose.dev.yml"
  ".github/workflows/ci-mastering.yml"
  "mastering/backend/Dockerfile"
  ".github/PULL_REQUEST_TEMPLATE.md"
)

for p in "${INCLUDE[@]}"; do
  if [ -e "$ROOT_DIR/$p" ]; then
    echo "Copying $p"
    mkdir -p "$(dirname "$BUNDLE_DIR/$p")"
    cp -a "$ROOT_DIR/$p" "$BUNDLE_DIR/$p"
  else
    echo "Warning: $p not found in repo; skipping"
  fi
done

# Add runbook and acceptance checklist
cat > "$BUNDLE_DIR/RUNBOOK.md" <<'MD'
BLACKGANG Mastering — Execution Runbook

What is inside this bundle:
- mastering/ (backend + frontend code)
- docker-compose.dev.yml (dev compose)
- .github/workflows/ci-mastering.yml (CI workflow, may be absent if not present)
- mastering/backend/Dockerfile
- .github/PULL_REQUEST_TEMPLATE.md (optional)

Quick start (preconditions):
- Git clone of the repo checked out to the working copy
- Python 3.11+, Node 18+, git, docker, docker-compose installed
- gh CLI installed & authenticated (for PR creation)

Agent Responsibilities (one-time):
1) Review files in this bundle for sensitive content / secrets.
2) Run the provided run_agent.sh script to create branch, push files, and open PR.
3) Execute smoke tests described in CHECKLIST.md and document results.

MD

cat > "$BUNDLE_DIR/CHECKLIST.md" <<'MD'
Acceptance checklist (minimum):

- [ ] Branch feature/mastering-engine created and pushed
- [ ] CI workflow present at .github/workflows/ci-mastering.yml
- [ ] Dockerfile present at mastering/backend/Dockerfile
- [ ] docker-compose.dev.yml exists and can bring up backend+frontend locally
- [ ] Backend: uvicorn main:app runs and responds on /health
- [ ] Upload a sample WAV (sample.wav) via /master or frontend and ensure a playable master is returned
- [ ] Check final integrated LUFS within ±0.7 dB of target (-10 LUFS)
- [ ] Confirm True Peak <= -1.0 dBTP
- [ ] PR opened against main with the specified title and description

Notes:
- If pedalboard/matchering fail to install, proceed with fallbacks but document differences.
MD

# run_agent.sh — the script the executing agent should run in the repo root
cat > "$BUNDLE_DIR/run_agent.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
# run_agent.sh
# Run this from the repository root. The script will:
# - create feature branch (feature/mastering-engine)
# - add any missing bundle files (copied into the bundle)
# - commit and push the branch
# - create a PR via gh

BRANCH="feature/mastering-engine"
PR_TITLE="feat(mastering): add one-click mastering engine (FastAPI backend + Next.js frontend)"
PR_BODY="Adds a server-side mastering pipeline (FastAPI) and demo Next.js frontend, plus Dockerfile, docker-compose.dev, and CI workflow. See mastering/README.md for running instructions. Caveats: requires ffmpeg and libsndfile; pedalboard / matchering optional and may need native deps. Please review the dsp_pipeline implementation and safety checks."

# ensure we're in a git repo
if [ ! -d .git ]; then
  echo "This script must be run from the root of the repository (where .git is located)."
  exit 1
fi

# Create branch
git fetch origin
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "Branch $BRANCH already exists locally. Checking it out."
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Copy bundle files from tools bundle location (this script is intended to be run with the bundle extracted into repo root under 'bundle_contents')
BUNDLE_SRC="./bundle_contents"
if [ -d "$BUNDLE_SRC" ]; then
  echo "Copying bundle contents into repo..."
  rsync -a "$BUNDLE_SRC/" ./
else
  echo "bundle_contents not found — ensure you've extracted the bundle into the repository root as ./bundle_contents"
  echo "Alternatively, run the make_execution_bundle.sh script to produce the zip and extract it in the repo root."
fi

# Add files that are likely new
git add .github/workflows/ci-mastering.yml mastering/backend/Dockerfile docker-compose.dev.yml .github/PULL_REQUEST_TEMPLATE.md || true

# Commit if there are changes
if git diff --staged --quiet; then
  echo "No changes to commit."
else
  git commit -m "chore(mastering): add CI, Dockerfile, docker-compose and PR template for mastering feature" || true
fi

# Push
git push -u origin "$BRANCH"

# Create PR using gh (requires authentication)
if command -v gh >/dev/null 2>&1; then
  gh pr create --base main --head "$BRANCH" --title "$PR_TITLE" --body "$PR_BODY"
else
  echo "gh CLI not found. Please open a PR manually from branch $BRANCH to main with title: $PR_TITLE"
fi

echo "Run complete. Please verify the PR on GitHub and run the CHECKLIST.md steps."
SH
chmod +x "$BUNDLE_DIR/run_agent.sh"

# generate a short sample WAV (1 second 440Hz sine) using Python stdlib
cat > "$BUNDLE_DIR/generate_test_tone.py" <<'PY'
#!/usr/bin/env python3
# generate_test_tone.py — simple 1s 44.1kHz 16-bit mono sine tone for smoke tests
import wave, struct, math
sr = 44100
dur = 1.0
freq = 440.0
n = int(sr * dur)
with wave.open('sample_test_tone.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    for i in range(n):
        t = i / sr
        val = int(32767 * 0.3 * math.sin(2 * math.pi * freq * t))
        wf.writeframes(struct.pack('<h', val))
print('Wrote sample_test_tone.wav')
PY

# run the generator and move sample into bundle
( cd "$BUNDLE_DIR" && python3 generate_test_tone.py || true )
if [ -f "$BUNDLE_DIR/sample_test_tone.wav" ]; then
  echo "Sample WAV created"
else
  # some environments wrote to current dir; try to move
  if [ -f "sample_test_tone.wav" ]; then
    mv sample_test_tone.wav "$BUNDLE_DIR/"
  fi
fi

# Add a README for the bundle
cat > "$BUNDLE_DIR/README.md" <<'MD'
Execution bundle for BLACKGANG mastering feature

Contents:
- run_agent.sh — script the executing agent may run from the repository root to commit files and open a PR
- RUNBOOK.md — operational runbook and notes
- CHECKLIST.md — acceptance checks
- sample_test_tone.wav — 1s sine tone for smoke testing

How to use:
1. Extract this bundle into the repository root (the script expects bundle_contents/ to exist)
2. Run: ./bundle_contents/run_agent.sh
3. Follow post-PR verification steps in CHECKLIST.md
MD

# Create bundle_contents directory for easy extraction
mkdir -p "$WORKDIR/bundle_archive"
cp -a "$BUNDLE_DIR" "$WORKDIR/bundle_archive/bundle_contents"

# Zip up
cd "$WORKDIR/bundle_archive"
zip -r "$OUT_ZIP" bundle_contents > /dev/null

# Move ZIP to repo root
mv "$OUT_ZIP" "$ROOT_DIR/"

# Cleanup
rm -rf "$WORKDIR"

echo "Created $OUT_ZIP"
`, 