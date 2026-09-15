.PHONY: validate-experiment validate-all verify-source explore sample-generalization validate-sample-generalization fixed-cohort-budget validate-fixed-cohort-budget dashboard

PYTHON ?= python3
CONFIG ?= configs/experiments/milestone_1_oracle_comparison.json
DATASET_ROOT ?= ml-latest
RUN_ID ?= exploration
OUTPUT_ROOT ?= artifacts
CLUSTER_MAP ?=
METADATA_ROOT ?= results/explorations
SUPPORT_PROBE_USERS ?=
SAMPLE_SCHEMES ?= uniform_user,activity_stratified_user,uniform_interaction,within_user_history
SAMPLE_FRACTIONS ?= 0.01,0.025,0.05,0.1,0.25,0.5,1.0
SAMPLE_REPLICATES ?= 10
SAMPLE_PANEL_SIZE ?= 2000
DASHBOARD_PORT ?= 8787
DASHBOARD_HOST ?= 127.0.0.1
SAMPLE_MODELS ?= popularity,rating_weighted_popularity,item_item_cosine
SAMPLE_SUPPORT_THRESHOLD ?= 20
SAMPLE_ITEM_ITEM_TOP_K ?= 100
SAMPLE_OUTPUT_ROOT ?= results/explorations/sample_generalization_full
SAMPLE_CACHE_ROOT ?= artifacts/sample_generalization_full_cache
FIXED_COHORT_BUDGET_OUTPUT ?= results/explorations/fixed_cohort_budget_v2

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

sample-generalization:
	$(PYTHON) scripts/run_sample_generalization.py \
		--schemes $(SAMPLE_SCHEMES) \
		--fractions $(SAMPLE_FRACTIONS) \
		--replicates $(SAMPLE_REPLICATES) \
		--panel-size $(SAMPLE_PANEL_SIZE) \
		--models $(SAMPLE_MODELS) \
		--support-threshold $(SAMPLE_SUPPORT_THRESHOLD) \
		--item-item-top-k $(SAMPLE_ITEM_ITEM_TOP_K) \
		--output $(SAMPLE_OUTPUT_ROOT) \
		--cache-dir $(SAMPLE_CACHE_ROOT)

validate-sample-generalization:
	$(PYTHON) scripts/validate_sample_generalization.py

fixed-cohort-budget:
	$(PYTHON) scripts/run_fixed_cohort_budget.py \
		--config configs/experiments/fixed_cohort_budget_v2.json \
		--output $(FIXED_COHORT_BUDGET_OUTPUT)

validate-fixed-cohort-budget:
	$(PYTHON) scripts/validate_fixed_cohort_budget.py \
		--config configs/experiments/fixed_cohort_budget_v2.json \
		--result-root $(FIXED_COHORT_BUDGET_OUTPUT)

validate-all: validate-experiment validate-sample-generalization validate-fixed-cohort-budget

dashboard:
	$(PYTHON) scripts/dashboard_server.py --host $(DASHBOARD_HOST) --port $(DASHBOARD_PORT)
