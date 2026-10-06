PY ?= .venv/bin/python

.PHONY: setup data test quick experiments demo writeup all

setup:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/generate_data.py

test:
	$(PY) -m pytest -q tests

quick:
	$(PY) scripts/run_experiments.py --quick

experiments:
	$(PY) scripts/run_experiments.py

demo:
	$(PY) scripts/demo.py

writeup:
	$(PY) docs/build_writeup.py

all: data test experiments writeup
