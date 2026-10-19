.PHONY: help lint format test test-all

help:
	@echo "make test      CPU smoke tests (about 1-2 min)"
	@echo "make test-all  smoke tests on all six backbones"
	@echo "make lint      check formatting, imports and style"
	@echo "make format    auto-format with isort and black"

test:
	python tests/smoke_test.py

test-all:
	python tests/smoke_test.py --all

lint:
	isort --check-only --diff .
	black --check --diff .
	flake8 .

format:
	isort .
	black .
