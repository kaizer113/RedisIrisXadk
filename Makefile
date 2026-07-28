.PHONY: install dev dev-norlings test lint dataset seed setup-context setup-iris setup-memory-bank seed-scale-memory reset-demo deploy deploy-all deploy-vm check-gcp configure-secrets

EXPERIENCE ?= valuewholesale
ENV_FILE ?= .env
PORT ?= 8080

install:
	uv sync --all-extras

dev:
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) uvicorn valuewholesale_agent.api:app --reload --port $(PORT)

dev-norlings:
	$(MAKE) dev EXPERIENCE=norlings ENV_FILE=.env.norlings PORT=8081

test:
	uv run pytest

lint:
	uv run ruff check .

dataset:
	uv run python -m scripts.generate_dataset --experience $(EXPERIENCE)

seed:
	$(MAKE) dataset
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.seed_redis

setup-context:
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.setup_context_retriever --env-file $(ENV_FILE)

setup-iris: seed setup-context

setup-memory-bank: dataset
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.create_memory_bank --env-file $(ENV_FILE)
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.seed_managed_memories --env-file $(ENV_FILE)

seed-scale-memory:
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.seed_scale_memories --yes

reset-demo:
	EXPERIENCE_ID=$(EXPERIENCE) uv run --env-file $(ENV_FILE) python -m scripts.reset_demo --yes

check-gcp:
	./scripts/check_gcp.sh

deploy:
	EXPERIENCE_ID=$(EXPERIENCE) VALUEWHOLESALE_CLOUD_RUN_ENV_FILE=$(ENV_FILE) ./scripts/deploy_gcp.sh

deploy-all: setup-iris setup-memory-bank deploy

deploy-vm: setup-iris setup-memory-bank
	EXPERIENCE_ID=$(EXPERIENCE) VALUEWHOLESALE_VM_ENV_FILE=$(ENV_FILE) ./scripts/deploy_vm.sh

configure-secrets:
	EXPERIENCE_ID=$(EXPERIENCE) VALUEWHOLESALE_CLOUD_RUN_ENV_FILE=$(ENV_FILE) ./scripts/configure_gcp_secrets.sh
