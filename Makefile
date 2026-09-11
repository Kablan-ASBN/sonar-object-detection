PY ?= .venv/bin/python
PIP := $(PY) -m pip

.PHONY: help setup test lint fmt audit splits ablation clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup: ## create .venv and install the package with dev extras
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,plots]"

test: ## run the test suite
	$(PY) -m pytest -q

lint: ## static checks
	$(PY) -m ruff check src/sonar tests

fmt: ## apply safe autofixes
	$(PY) -m ruff check --fix src/sonar tests
	$(PY) -m ruff format src/sonar tests

audit: ## fail if any training split overlaps any evaluation split
	$(PY) -m sonar.cli audit leakage \
		--root raw=data/line2voc \
		--root denoised=data/line2voc_preprocessed \
		--root augmented=data/line2voc_preprocessed_augmented

splits: ## regenerate one seeded split and copy it to every dataset root
	$(PY) -m sonar.cli split \
		--root data/line2voc \
		--also-root data/line2voc_preprocessed \
		--also-root data/line2voc_preprocessed_augmented \
		--seed 42

ablation: ## reproduce the leakage/flip ablation (CPU, a few hours)
	$(PY) experiments/ablation.py --out experiments/results/ablation.json

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
