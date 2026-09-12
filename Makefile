.PHONY: validate-experiment verify-source explore dashboard

PYTHON ?= python3
CONFIG ?= configs/experiments/milestone_1_oracle_comparison.json
DATASET_ROOT ?= ml-latest
RUN_ID ?= exploration
OUTPUT_ROOT ?= artifacts
CLUSTER_MAP ?=
METADATA_ROOT ?= results/explorations
SUPPORT_PROBE_USERS ?=
DASHBOARD_HOST ?= 127.0.0.1
DASHBOARD_PORT ?= 8787

EXPLORATION_ARGS = \
	--config $(CONFIG) \
	--run-id $(RUN_ID) \
	--output-root $(OUTPUT_ROOT) \
	--metadata-root $(METADATA_ROOT)

ifneq ($(strip $(CLUSTER_MAP)),)
EXPLORATION_ARGS += --cluster-map $(CLUSTER_MAP)
endif
ifneq ($(strip $(SUPPORT_PROBE_USERS)),)
EXPLORATION_ARGS += --support-probe-users $(SUPPORT_PROBE_USERS)
endif

validate-experiment:
	$(PYTHON) scripts/separate_dataset.py --verify
	$(PYTHON) scripts/validate_experiment.py

verify-source:
	$(PYTHON) scripts/validate_experiment.py --dataset-root $(DATASET_ROOT) --verify-files

explore: validate-experiment
	$(PYTHON) scripts/explore_dataset.py $(EXPLORATION_ARGS)

dashboard:
	$(PYTHON) scripts/dashboard_server.py --host $(DASHBOARD_HOST) --port $(DASHBOARD_PORT)
