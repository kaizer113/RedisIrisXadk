# ValueHarbor architecture

ValueHarbor is a fictional membership-warehouse shopping agent built to demonstrate how
Google Agent Development Kit (ADK) and Redis IRIS services can work together in an ecommerce
journey. The application is a FastAPI service with a browser chat UI. The current public demo
runs on a Compute Engine VM in `us-east4-c`; Cloud Run remains an optional deployment target.

![ValueHarbor system architecture](docs/architecture.svg)

The editable Mermaid source is [`docs/architecture.mmd`](docs/architecture.mmd).

## Components

| Layer | Component | Responsibility |
|---|---|---|
| Client | Browser chat UI | Sends a member message, selected model, member ID, and session ID; renders the streamed answer and live trace. |
| Application | FastAPI | Exposes the public UI and API, validates requests, runs parallel prefetches, applies the cache policy, and streams newline-delimited JSON events. |
| Agent runtime | Google ADK `Runner` | Runs the Vale agent, manages a session, invokes tools, calls Gemini, and triggers post-turn memory promotion. |
| Models | Gemini 2.5 Flash / Gemini 2.5 Pro | Flash is the fast default; Pro is the slower, heavier reasoning option. The selected model chooses one of two prebuilt runners. |
| Commerce data | Redis database + Query Engine | Stores the checked-in catalog, policies, inventory, member, order, and cart data; supports lexical and optional vector product retrieval. |
| Governed context | Redis Context Retriever | Exposes live member, inventory, and order entities through a governed tool surface discovered and called by the agent. |
| Semantic cache | Redis LangCache | Serves semantically similar public policy answers without invoking ADK or Gemini. Personalized requests are not cache eligible. |
| Redis memory | Redis Agent Memory | Receives explicit user and assistant session events and stores/retrieves durable member preference memories. |
| Google sessions | Agent Platform Sessions | Persists ADK session events when `VertexAiSessionService` is configured. Sessions are visible under Agent Platform in the GCP console. |
| Google memory | ADK Memory Bank | Retrieves long-term memories and generates new memories from the completed ADK session after a model-generated turn. |
| Deployment | Compute Engine / Cloud Run | The current public demo runs on an `e2-standard-4` VM; Artifact Registry and Cloud Build build and store the image. |

## One chat request

`POST /api/chat/stream` is the main demo path.

1. FastAPI normalizes the member and session IDs and determines whether the prompt is a
   non-personalized public-policy question eligible for LangCache.
2. Four reads start concurrently:
   - Redis LangCache, when eligible;
   - recent Redis Agent Memory session events;
   - Redis Agent Memory long-term memories;
   - ADK Memory Bank long-term memories.
3. Each completed read is emitted to the UI as a trace step with client-observed latency and
   retrieved snippets. Redis receives the user event independently of the ADK session backend.
4. On a LangCache hit, the cached answer is returned immediately. The ADK runner, Gemini,
   Agent Platform Session update, tool calls, and Memory Bank promotion are skipped. Redis
   Agent Memory still receives both the user and cached assistant events.
5. On a cache miss or bypass, the results from both memory systems are added to ADK state and
   the runner corresponding to the selected model processes the turn.
6. ADK may invoke catalog, policy, cart, memory, or Context Retriever tools. Tool start,
   completion, result summary, and elapsed time are streamed to the UI.
7. ADK stores the conversational turn through its shared session service. The agent's
   post-turn callback asks ADK to generate Memory Bank memories from that session.
8. FastAPI records the assistant event in Redis Agent Memory and stores an eligible public
   policy answer in LangCache.

Only reads are benchmarked in the demo. Memory and cache writes are intentionally outside the
reported comparison because this is a short workshop and cross-system write consistency is not
a goal.

## Model selection and session sharing

The process creates one ADK `Runner` for each approved model:

- `gemini-2.5-flash` — fast/default;
- `gemini-2.5-pro` — heavier reasoning.

Both runners receive the same `session_service` object, `memory_service` object, ADK app name,
member ID, and session ID. Switching the model in the chat therefore does **not** create a
separate conversation: both models continue the same session and see the same session state.

With managed configuration, the shared services are `VertexAiSessionService` and
`VertexAiMemoryBankService`. Without a configured Agent Engine ID, local development falls back
to process-local `InMemorySessionService` and `InMemoryMemoryService`. The two local runners
still share those same in-process instances, but all local session and memory contents disappear
when the process restarts and are not visible in the GCP console.

## Session and long-term memory paths

The systems overlap deliberately but are not interchangeable.

| Concern | Redis path | Google ADK path |
|---|---|---|
| Short-term conversation | FastAPI explicitly writes user and assistant events to Redis Agent Memory. Each request explicitly reads recent events. | The selected runner reads and appends the ADK session through `VertexAiSessionService`, or through the local in-memory fallback. |
| Long-term memory | Explicit preferences are written to Redis Agent Memory; semantic recall runs before each generated turn. | After a generated turn, the callback promotes the ADK session to Memory Bank; semantic recall runs before the next generated turn. |
| Independence | Redis event writes continue regardless of which ADK session service is selected. | Replacing `InMemorySessionService` with `VertexAiSessionService` changes ADK persistence, not Redis writes. |
| Console visibility | Inspect with Redis Cloud/Redis Insight and the Agent Memory service. | Managed sessions and memories appear under Agent Platform for the configured Agent Engine and region. In-memory fallbacks do not. |

The comparison endpoint, `POST /api/memory/compare`, sends the same query and member scope to
Redis Agent Memory and ADK Memory Bank concurrently. It reports client-observed read latency,
median latency over repeated runs, `precision@k`, and `recall@k` against optional expected terms.
The supplied evaluation cases in `data/generated` make the accuracy comparison reproducible.

## Data and retrieval

The canonical demo dataset is checked into `data/generated` as deterministic JSONL. This makes
the workshop reproducible even though the online services are managed and mutable.

- `scripts/generate_dataset.py` regenerates the local fixtures.
- `scripts/seed_redis.py` loads commerce entities and creates Redis search indexes.
- `scripts/setup_context_retriever.py` imports governed entities and configures the Context
  Retriever surface.
- `scripts/create_memory_bank.py` creates or updates the regional ADK Memory Bank.
- `scripts/seed_managed_memories.py` seeds equivalent facts in Redis Agent Memory and ADK
  Memory Bank for comparison.

Product discovery uses Redis Query Engine. Lexical retrieval always works; optional semantic
retrieval uses `text-embedding-005` embeddings with a 768-dimensional HNSW cosine index.
Context Retriever is the required path for live member, warehouse inventory, and order data in
the agent instructions. Redis fixtures remain available as a local-development fallback.

## Deployment and configuration

The deployment target is GCP project `central-beach-194106` (`redislabs-sales-project`) in
`us-east4`. Resources that support labels receive:

```text
owner=lionel_giavelli,app=valueharbor,environment=demo
```

`scripts/deploy_vm.sh` enables the required APIs, builds the image with Cloud Build, stores it
in Artifact Registry, and updates the public `valueharbor-demo` Compute Engine VM. The VM uses
Premium Tier networking, gVNIC, and a dedicated firewall tag exposing only TCP port 80. The
optional `scripts/deploy_gcp.sh` path deploys the same image to Cloud Run. Runtime configuration
is supplied through environment variables. The local `.env` is ignored by Git; `.env.example`
documents the supported names without credentials. Production deployments should bind secrets
through Secret Manager and use a dedicated least-privilege service identity.

The Redis endpoint is currently public and is expected to move to private connectivity later.
That network change should reduce Redis round-trip latency without changing the application
component boundaries or request flow shown here.

## Failure behavior

- Missing Redis database configuration uses deterministic local catalog and cart fixtures.
- Missing managed Agent Platform configuration uses in-process ADK session and memory services.
- Managed memory and post-turn promotion fail open so shopping can continue during a workshop.
- Personalized requests bypass LangCache to avoid serving another member's response.
- A model timeout produces an error trace instead of an unbounded request.

These fallbacks are for demo resilience, not a production consistency or availability design.

## Code map

| Path | Purpose |
|---|---|
| `valueharbor_agent/api.py` | HTTP API, parallel prefetch, cache branching, runners, streaming trace, and memory comparison endpoint. |
| `valueharbor_agent/agent.py` | Vale's instructions, model-specific agent construction, and Memory Bank promotion callback. |
| `valueharbor_agent/tools.py` | ADK commerce, Context Retriever, cart, and memory tools. |
| `valueharbor_agent/services.py` | Redis, LangCache, Redis Agent Memory, Context Retriever, and Vertex Memory Bank adapters. |
| `valueharbor_agent/config.py` | Environment-driven service, model, project, and region configuration. |
| `valueharbor_agent/static/` | Browser chat UI and live execution trace. |
| `data/generated/` | Versioned, reproducible demo entities, memories, and retrieval evaluation cases. |
| `scripts/` | Dataset, managed-service setup, GCP deployment, and secret configuration. |
