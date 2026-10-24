PY ?= python
export PYTHONPATH := src

.PHONY: install data reproduce reproduce-main seeds aggregate smoke test lint scan models demo app docker clean

install:            ## install pinned dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:               ## download PhiUSIIL + PhreshPhish URL columns and verify checksums
	$(PY) scripts/download_data.py
	$(PY) scripts/download_phreshphish.py

reproduce: reproduce-main seeds aggregate   ## everything in results/ (about 3 h on 8 CPU cores, no GPU)

reproduce-main: data   ## seed 42 on both datasets; also saves the demo models to models/
	$(PY) -m phishadv.experiment --config phresh --save-models models/phreshphish
	$(PY) -m phishadv.experiment --config full --save-models models/phiusiil

seeds: data         ## seeds 43 and 44 on both datasets -> results/seeds/
	for s in 43 44; do \
	  $(PY) -m phishadv.experiment --config phresh --seed $$s && \
	  $(PY) -m phishadv.experiment --config full --seed $$s || exit 1; \
	done

aggregate:          ## cross-seed / cross-dataset tables and headline chart
	$(PY) -m phishadv.aggregate

smoke:              ## fast end-to-end runs on the bundled 1,500-URL samples -> results-smoke/
	$(PY) -m phishadv.experiment --config smoke
	$(PY) -m phishadv.experiment --config smoke-phresh

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts app

scan:               ## secret scan (must exit 0 before every push)
	gitleaks detect --source . --no-banner

models:             ## train + save demo models only (PhreshPhish), without the full experiment
	$(PY) scripts/download_phreshphish.py
	$(PY) -m phishadv train --dataset phreshphish --out models/phreshphish

demo:               ## score one URL: make demo URL='http://example.com/login'
	$(PY) -m phishadv score "$(URL)" --models-dir models/phreshphish

app:                ## local Streamlit demo on http://localhost:8501 (pip install -r requirements-demo.txt)
	$(PY) -m streamlit run app/streamlit_app.py --server.address localhost --server.headless true

docker:             ## build the image and run the smoke test + unit tests inside it
	docker build -t phishadv .
	docker run --rm phishadv

clean:
	rm -rf results-smoke .pytest_cache .ruff_cache
