# ValueHarbor Shopping Agent

ValueHarbor is a fictional membership-warehouse ecommerce demo. It recreates the core ideas in the Redis IRIS workshop as one end-to-end shopping journey, implemented with Google Agent Development Kit (ADK) and deployable to Compute Engine or Cloud Run.

The agent can discover products, compare member pricing, check warehouse inventory, inspect recent orders, answer grounded policy questions, build a cart, and remember household shopping preferences.

## Quick links

- [Recommended demo flow](docs/demo.md)
- [Architecture and request flow](ARCHITECTURE.md)
- [Reproducible dataset](data/README.md)
- Current public demo: [http://34.182.213.82](http://34.182.213.82)

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete component map, request flow,
session and long-term memory paths, deployment topology, and a rendered system diagram.

| Capability | Service | Demo role |
|---|---|---|
| Agent runtime | Google ADK + Gemini on Vertex AI | Tool selection and response generation |
| Catalog and policies | Redis database + Query Engine/vector search | Low-latency grounded retrieval |
| Live commerce context | Redis Context Retriever | Governed access to inventory/order entities |
| Cache routing | RedisVL Semantic Router | Safe semantic classification of reusable public-policy prompts |
| Response cache | Redis LangCache | Cache-aside for non-personalized policy answers |
| Memory A | Redis Agent Memory | Session events and scoped semantic/episodic preferences |
| Memory B | Vertex AI ADK Memory Bank | ADK-managed conversation memory |
| Hosting | Compute Engine / Cloud Run | Public web app and API in `us-east4` |

The deployed Compute Engine workload and Agent Platform Memory Bank are colocated in `us-east4`.
When an Agent Engine ID is configured, ADK events are stored in Agent Platform Sessions while
Redis Agent Memory continues receiving its independent short-term event stream. The chat offers
two server-approved model choices: Gemini 2.5 Flash for speed and Gemini 2.5 Pro for heavier
reasoning.

## Live agent trace

Every shopping request runs RedisVL semantic routing alongside Redis Agent Memory session
history, Redis long-term memory, and Vertex ADK Memory Bank retrieval. Safe public-policy route
matches then check LangCache before ADK runs. The web UI streams those steps live, expands
retrieved facts inline, times every ADK and Context Retriever tool call, and finishes with
orchestration/generation and total request latency.

`make setup-memory-bank` idempotently creates or updates the named Vertex Memory Bank, saves
its non-secret resource ID in `.env`, and seeds the same checked-in facts into both managed
long-term memory providers. New conversation turns continue to feed Redis session memory and
ADK Memory Bank generation independently.

## Local start

```bash
cp .env.example .env
uv sync --all-extras
uv run uvicorn valueharbor_agent.api:app --env-file .env --reload --port 8080
```

`.env.example` is the checked-in configuration template. Each developer copies it to the gitignored `.env` file and adds their own service endpoints, IDs, and credentials there. Never commit a live Redis URL or API key.

Open [http://localhost:8080](http://localhost:8080). Without Redis credentials, the catalog, warehouse, member, and cart tools use deterministic fixtures; managed IRIS capabilities show as unconfigured.

## Configure managed services

Fill the corresponding variables in your local `.env`:

- `REDIS_URL`
- `MCP_AGENT_KEY`
- `LANGCACHE_HOST`, `LANGCACHE_CACHE_ID`, `LANGCACHE_API_KEY`
- `AGENT_MEMORY_BASE_URL`, `AGENT_MEMORY_STORE_ID`, `AGENT_MEMORY_API_KEY`
- `GOOGLE_AGENT_ENGINE_ID`

Then seed the Redis catalog and inventory:

```bash
make setup-iris
```

`make setup-iris` is the repeatable Redis setup command. It regenerates the dataset, seeds the Redis database, creates or updates the `ValueHarbor Shopping` Context Surface through `ctxctl`, imports the entities, and creates a surface-scoped agent key when `.env` does not already contain one.

Once every managed-service ID is present in `.env` and GCP authentication is active, `make deploy-all` performs the Redis setup and deploys the public Cloud Run service in one command. This demo path uses the project's existing default Cloud Run identity and revision environment variables, so it does not create service accounts or change project IAM. Production deployments should use a dedicated least-privilege service identity and Secret Manager.

If Cloud Run public-invoker permissions are unavailable, `make deploy-vm` deploys the complete UI and API to a public `e2-standard-4` VM in `us-east4-c`. The VM uses Premium Tier networking, gVNIC, a balanced persistent disk, and a firewall tag that exposes only HTTP port 80. The command prints the public IP URL when its health check succeeds.

## Redeploy the public demo

The checked-in deployment path is idempotent and does not add project IAM bindings. From a
machine with `gcloud`, `uv`, and access to the target GCP project:

```bash
cp .env.example .env                 # first deployment only; add your service credentials
gcloud auth login
gcloud auth application-default login
gcloud config set project central-beach-194106
make deploy-vm
```

`make deploy-vm` regenerates and seeds the demo data, updates Context Retriever, creates or
reuses the ADK Memory Bank, seeds both long-term memory providers, builds the container, and
updates the existing `valueharbor-demo` VM in `us-east4-c`. It prints the public URL after the
health check passes.

For a code-only redeploy that leaves the managed-service data unchanged:

```bash
./scripts/deploy_vm.sh
```

Verify the deployment:

```bash
curl -fsS http://34.182.213.82/api/health
```

The expected response reports both Gemini models and all Redis/Google integrations as `true`.
Stop the VM when the demo is not needed:

```bash
gcloud compute instances stop valueharbor-demo --zone us-east4-c
```

The deterministic JSONL dataset lives in [`data/generated`](data/generated) and includes products, warehouses, inventory, members, normalized orders, policies, identical memory seeds, and labeled retrieval-evaluation cases. See [`data/README.md`](data/README.md) for its schema and Redis key model.

The product index uses HNSW, 768-dimensional `FLOAT32` vectors, and cosine distance to match Vertex `text-embedding-005`. Category filtering is applied before KNN search.

Vector embedding is off by default so the dataset can be recreated without GCP credentials. After application-default authentication is available, set `VALUEHARBOR_VECTOR_SEARCH_ENABLED=true` and rerun `make seed` to populate product embeddings; lexical Redis search remains available either way.

## GCP deployment

The target project is `central-beach-194106` (display name `redislabs-sales-project`). Resource labels use `owner=lionel_giavelli`, plus app/environment labels where supported.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=central-beach-194106
make check-gcp

# Memory Bank is colocated with the application.
uv run python scripts/create_memory_bank.py --project "$GOOGLE_CLOUD_PROJECT" --location us-east4
export GOOGLE_AGENT_ENGINE_ID=<id printed above>

make deploy
```

The Cloud Run service is public by default and the deploy command prints its HTTPS endpoint URL. Set `PUBLIC_ACCESS=false` only when you intentionally need a private deployment.

The Cloud Run deploy script enables required APIs, reuses the project's existing runtime identity,
creates a labeled Artifact Registry repository, builds the image, and deploys a labeled service in
`us-east4`. It does not add project IAM bindings.

After the Redis services are provisioned, export their values in your shell and run:

```bash
make configure-secrets
```

That script creates or versions labeled Secret Manager secrets without putting secret values in command arguments, then binds them to Cloud Run. Service endpoints and IDs are configured as ordinary environment variables.

Some GCP resource types, including service accounts and API enablement records, do not support user labels. The owner label is applied to every created resource type that supports labels.

## Workshop flow

1. Run product discovery and warehouse inventory using fixtures.
2. Seed Redis and repeat the same grounded queries.
3. Connect Context Retriever and inspect its live tool schemas.
4. Ask the same public policy question twice to observe LangCache.
5. Save a shopping preference and watch Redis session memory, both long-term memory systems,
   governed tool calls, and total generation latency appear in the live trace.

## Quality checks

```bash
make lint
make test
```
