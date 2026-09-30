PYTHON ?= python3
Q ?= What are the technical evaluation gates and their target thresholds?

.PHONY: help install install-optional test audit index baseline eval replay controller ablate decompose ablate3 full ablate4 coverage replay-llm replay-offline llm-check demo all ablate-llm stream schema manifest lock clean

help:
	@echo "all               one-command clean run: venv + pinned deps + tests + indexes + full replay"
	@echo "install           install pinned runtime + dev dependencies"
	@echo "install-optional  install model-based backends (torch; not tested in Phase 1)"
	@echo "test              run the test suite"
	@echo "audit             print corpus audit (docs, sections, chunk lengths, duplicates)"
	@echo "index             build BM25 + dense indexes and write indexes/chunks.jsonl"
	@echo "baseline          answer Q=\"...\" with the non-streaming baseline"
	@echo "eval              run the baseline over eval/dev_scenarios and print metrics"
	@echo "replay            full replay suite, gates G1-G6 -> results/replay/ (current PRISM_* profile)"
	@echo "replay-llm        full replay with the intended local 7-8B Ollama model (LLM_MODEL=...)"
	@echo "replay-offline    full replay with the offline CPU profile (no model)"
	@echo "demo              wall-clock demo of the four storyboard scenarios (reports/DEMO_SCRIPT.md)"
	@echo "ablate-llm        fill the LLM rows of the phase 2-4 ablations (needs the 7-8B Ollama model)"
	@echo "llm-check         verify the Ollama server and model (reachable, pulled, parameter size)"
	@echo "controller        G2 controller metrics over eval/dev_scenarios (phase 2)"
	@echo "ablate            controller ablation + threshold grid -> reports/PHASE_2_CONTROLLER_ABLATION.md"
	@echo "decompose         G3 multi-intent identification + fusion metrics (phase 3)"
	@echo "ablate3           phase 3 ablations -> reports/PHASE_3_ABLATION.md"
	@echo "full              G4 grounding + G5 session refinement over the dev scenarios (phase 4)"
	@echo "ablate4           grounding-judge ablation -> reports/PHASE_4_GROUNDING_ABLATION.md"
	@echo "coverage          G6 telemetry trace coverage of logs/telemetry.jsonl"
	@echo "stream            stream U=\"frag 1 | frag 2\" through controller + decomposition (demo)"
	@echo "schema            regenerate schemas/*.schema.json from src/schemas.py"
	@echo "manifest          recompute data/corpus/MANIFEST.json hashes"

all:
	bash scripts/run_all.sh

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-optional:
	$(PYTHON) -m pip install -r requirements-optional.txt

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) -m src.corpus.audit

index:
	$(PYTHON) -m src.corpus.build_index

baseline:
	$(PYTHON) -m src.baseline --query "$(Q)"

eval:
	$(PYTHON) eval/run_eval.py --system baseline

controller:
	$(PYTHON) eval/run_eval.py --system controller_only --metrics g2

ablate:
	$(PYTHON) eval/ablate_controller.py --out reports/PHASE_2_CONTROLLER_ABLATION.md

decompose:
	$(PYTHON) eval/run_eval.py --system decompose_fusion --metrics g3

ablate3:
	$(PYTHON) eval/ablate_decompose.py --out reports/PHASE_3_ABLATION.md

full:
	$(PYTHON) eval/run_eval.py --system full --metrics g4,g5

ablate4:
	$(PYTHON) eval/ablate_grounding.py --out reports/PHASE_4_GROUNDING_ABLATION.md

coverage:
	$(PYTHON) -m src.telemetry.coverage logs/telemetry.jsonl

U ?= Which gate covers | early retrieval | and how is it validated?
stream:
	$(PYTHON) -m src.engine --utterance "$(U)"

# Full replay suite (phase 5): streaming system + baseline over the dev scenarios, gates G1-G6,
# results/replay/{results.json,summary.md,telemetry.jsonl}. Uses the PRISM_* profile in the environment.
replay:
	$(PYTHON) eval/replay.py --out results/replay

# Intended runtime profile: local Ollama 7-8B model (must be running and pulled; checked first).
LLM_MODEL ?= llama3.1:8b
LLM_ENV = PRISM_LLM_PROVIDER=ollama PRISM_LLM_MODEL=$(LLM_MODEL) PRISM_DECOMPOSER=llm PRISM_GROUNDING_JUDGE=llm
replay-llm:
	$(LLM_ENV) $(PYTHON) eval/replay.py --out results/replay-llm

llm-check:
	$(LLM_ENV) $(PYTHON) -m src.llm.client --check

demo:
	$(PYTHON) eval/replay.py --scenario single_02 --scenario multi_03 --scenario late_02 --scenario present_04 \
		--clock wall --show --out results/demo
	$(PYTHON) eval/show_turns.py results/demo/results.json

# Only the provider/model are set: the scripts keep their rule-based rows and add the LLM rows.
LLM_BASE_ENV = PRISM_LLM_PROVIDER=ollama PRISM_LLM_MODEL=$(LLM_MODEL)
ablate-llm:
	$(LLM_BASE_ENV) $(PYTHON) -m src.llm.client --check
	$(LLM_BASE_ENV) $(PYTHON) eval/ablate_controller.py --out reports/PHASE_2_CONTROLLER_ABLATION.md
	$(LLM_BASE_ENV) $(PYTHON) eval/ablate_decompose.py --out reports/PHASE_3_ABLATION.md
	$(LLM_BASE_ENV) $(PYTHON) eval/ablate_grounding.py --out reports/PHASE_4_GROUNDING_ABLATION.md

# Offline CPU profile, explicitly (no model needed).
OFFLINE_ENV = PRISM_LLM_PROVIDER=none PRISM_DECOMPOSER=rules PRISM_GROUNDING_JUDGE=lexical
replay-offline:
	$(OFFLINE_ENV) $(PYTHON) eval/replay.py --out results/replay-offline

schema:
	$(PYTHON) scripts/export_schemas.py

manifest:
	$(PYTHON) scripts/corpus_manifest.py --write

lock:
	uv lock

clean:
	rm -rf indexes results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
