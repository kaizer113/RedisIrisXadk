.PHONY: install dev test lint dataset seed setup-context setup-iris setup-memory-bank reset-demo deploy deploy-all deploy-vm check-gcp configure-secrets

install:
	uv sync --all-extras

dev:
	uv run uvicorn valueharbor_agent.api:app --env-file .env --reload --port 8080

test:
	uv run pytest

lint:
	uv run ruff check .

dataset:
	uv run python -m scripts.generate_dataset

seed:
	$(MAKE) dataset
	uv run python -m scripts.seed_redis

setup-context:
	uv run python -m scripts.setup_context_retriever

setup-iris: seed setup-context

setup-memory-bank: dataset
	uv run python scripts/create_memory_bank.py
	uv run python -m scripts.seed_managed_memories

reset-demo:
	uv run python -m scripts.reset_demo --yes

check-gcp:
	./scripts/check_gcp.sh

deploy:
	./scripts/deploy_gcp.sh

deploy-all: setup-iris setup-memory-bank deploy

deploy-vm: setup-iris setup-memory-bank
	./scripts/deploy_vm.sh

configure-secrets:
	./scripts/configure_gcp_secrets.sh
