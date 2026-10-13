# make demo   -> venv, deps, models, test packs, tests, Part 1 + Part 3 evals, then the app
# Same steps as `python run.py demo`; this file is for macOS/Linux (Windows: use run.py or Docker).
PYTHON ?= python3.11
PY := .venv/bin/python

.PHONY: demo app test e2e chat eval docker setup

$(PY):
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps basic-pitch==0.4.0
	$(PY) -m playwright install chromium

setup: $(PY)

demo: $(PY)
	$(PY) run.py demo

app: $(PY)
	$(PY) run.py app

test: $(PY)
	$(PY) run.py test

e2e: $(PY)
	$(PY) run.py e2e

chat: $(PY)
	$(PY) run.py chat

eval: $(PY)
	$(PY) run.py eval

docker:
	docker compose up --build
