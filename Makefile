PY ?= python
export PYTHONPATH := src

.PHONY: install data reproduce smoke test lint scan clean

install:            ## install pinned dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:               ## download PhiUSIIL and verify SHA-256 checksums
	$(PY) scripts/download_data.py

reproduce: data     ## full experiment -> results/ (about 20 min on 8 CPU cores, no GPU)
	$(PY) -m phishadv.experiment --config full

smoke:              ## fast end-to-end run on the bundled 1,500-URL sample -> results-smoke/
	$(PY) -m phishadv.experiment --config smoke

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts

scan:               ## secret scan (must exit 0 before every push)
	gitleaks detect --source . --no-banner

clean:
	rm -rf results-smoke .pytest_cache .ruff_cache
