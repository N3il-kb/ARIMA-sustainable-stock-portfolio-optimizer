.PHONY: setup clean test

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

clean:
	rm -rf venv
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf */__pycache__

test:
	venv/bin/python -m pytest
