PYTHON ?= python3
CONFIG ?= configs/default.yaml
GRU_CONFIG ?= configs/model_gru.yaml
MODEL ?= popularity
SPLIT ?= validation

.PHONY: test lint prepare-data prepare-demo run-baselines train evaluate demo

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

prepare-data:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --step make-dataset

prepare-demo:
	$(PYTHON) scripts/build_demo_artifacts.py
	$(PYTHON) scripts/build_category_mapping.py

run-baselines:
	$(PYTHON) scripts/run_baselines.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train_model.py --config $(GRU_CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate_model.py --config $(CONFIG) --model $(MODEL) --split $(SPLIT)

demo:
	$(PYTHON) scripts/run_demo.py
