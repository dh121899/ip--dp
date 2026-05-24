.PHONY: install test lint format smoke clean

PYTHON ?= .\.venv\Scripts\python.exe
RUFF ?= .\.venv\Scripts\ruff.exe
BLACK ?= .\.venv\Scripts\black.exe

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(RUFF) check .

format:
	$(BLACK) src tests
	$(RUFF) check --fix src tests

smoke: test lint

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'build', 'dist']]"

