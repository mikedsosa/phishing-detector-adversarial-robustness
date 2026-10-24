# Reproducible CPU environment. `docker build -t phishadv . && docker run --rm phishadv`
# runs the unit tests and the end-to-end smoke test on the bundled samples, then
# trains small demo models on the PhiUSIIL sample and scores one example URL.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=src
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home app
WORKDIR /home/app/phishadv

COPY requirements.txt requirements-dev.txt ./
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-dev.txt

COPY --chown=app:app . .
# WORKDIR is created root-owned; the smoke run writes results-smoke/ here.
RUN chown app:app /home/app/phishadv
USER app

CMD ["sh", "-c", "make test && make smoke PY=python && python -m phishadv train --dataset sample --out /tmp/models && python -m phishadv score 'http://secure-login.example-account.xyz/verify.php?id=1' --models-dir /tmp/models"]
