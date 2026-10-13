# Beat Agent: app + evals in one image (CPU only).
# bookworm pinned: Playwright's --with-deps installer targets Debian 12
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Python deps first so code changes don't reinstall torch.
# basic-pitch goes in with --no-deps (its default deps pull TensorFlow, which the ONNX model doesn't need).
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install --no-deps basic-pitch==0.4.0 \
 && python -m playwright install --with-deps chromium

COPY . .
# Beat This! checkpoint (8 MB) baked into the image.
RUN python run.py models

EXPOSE 8000
# Builds both synthetic test packs, runs the tests and the Part 1 / Part 3 evals, then serves the app.
CMD ["python", "run.py", "demo"]
