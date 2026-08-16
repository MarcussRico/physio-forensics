.PHONY: install test selftest demo clean

install:
	pip install -r requirements.txt

test:
	pytest -q

selftest:
	python -m physioforensics.cli selftest

# Full pipeline on the synthetic corpus: render -> features -> evaluate.
# Reproduces every number in the README with no dataset access required.
demo:
	python -m physioforensics.cli synth --out data/synthetic -n 20
	python -m physioforensics.cli features --root data/synthetic --out data/features.csv --roi fixed
	python -m physioforensics.cli evaluate --table data/features.csv --out data/results.csv
	python -m physioforensics.cli importance --table data/features.csv --out data/importance.csv

clean:
	rm -rf data models .pytest_cache __pycache__ physioforensics/__pycache__ tests/__pycache__
