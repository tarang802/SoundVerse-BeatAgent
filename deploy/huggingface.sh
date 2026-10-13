#!/usr/bin/env bash
# Publish Beat Agent as a Hugging Face Docker Space.
#
#   bash deploy/huggingface.sh <hf-username>/<space-name>
#
# Builds a clean copy of the committed code (git HEAD), swaps in a Space-specific
# Dockerfile (port 7860, runs as the Space's non-root user, starts the app directly
# instead of the full eval run), drops the listening MP3s (Hugging Face rejects plain-git
# binaries), and pushes it. Git asks for your HF username and a *write* token as password.
set -euo pipefail

SPACE="${1:?usage: bash deploy/huggingface.sh <hf-username>/<space-name>}"
OUT="$(mktemp -d)"
git archive HEAD | tar -x -C "$OUT"
cd "$OUT"

rm -rf evals/evidence .github
find . -name "*.mid" -o -name "*.mp3" | xargs rm -f

cat > Dockerfile <<'DOCKER'
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 PORT=7860 HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install --no-deps basic-pitch==0.4.0
COPY . .
# model + the synthetic library the app lists (A1-A7, P1-P4); Spaces run as uid 1000
RUN python run.py models && python run.py testpack && python run.py p3pack && chmod -R a+rwX /app
EXPOSE 7860
CMD ["python", "run.py", "app"]
DOCKER

{ printf -- '---\ntitle: Beat Agent\nemoji: 🎧\ncolorFrom: purple\ncolorTo: pink\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n'
  cat README.md; } > README.space && mv README.space README.md

if [ "${DRY_RUN:-}" = 1 ]; then echo "dry run: $OUT"; exit 0; fi

git init -q -b main
git add -A
git -c user.name="tarang802" -c user.email="176744070+tarang802@users.noreply.github.com" commit -qm "Beat Agent"
git push --force "https://huggingface.co/spaces/$SPACE" main
echo
echo "Pushed. Build log: https://huggingface.co/spaces/$SPACE  (first build ~15 min)"
