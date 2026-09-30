# CONFIGURATION ONLY in this repository's build environment: this image has not been built or
# run there (no Docker daemon). See reports/FINAL_CHECKLIST.md for what remains to verify.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PRISM_IN_CONTAINER=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

# One non-interactive command: build indexes from data/corpus/, run the full replay suite
# (streaming system + baseline over eval/dev_scenarios), write results/replay/{results.json,summary.md}
# and exit non-zero if any gate fails. The LLM profile comes from PRISM_* environment variables.
CMD ["sh", "-c", "python -m src.corpus.build_index && python eval/replay.py --out results/replay"]
