# Value Wholesale self-managed Iris GA setup

The Value Wholesale demo on port `8083` has its own application, dependency environment,
container, and configuration. It uses Agent Memory, published Agent Playbooks, Intent Router,
Context Gateway, Lens, and Agent Mesh. Its shopping workflow is a bounded Google GenAI tool loop
through Vertex AI. The default Value Wholesale, Norling's, and Value Travel demos keep their
existing runtimes and dependencies.

**Status, November 4, 2026:** the isolated GA application passes local validation. GKE and the
shared demo VM are running; RDI 2.2.1, monitoring, and the rebuilt synthetic SQL source have
passed live checks. A source-built suite 1.0.0 package is prepared from pinned official Redis
source with verified published dependencies. The issued suite license is still pending, so
GA service deployment, resource/key provisioning, and the port-8083 cutover have not occurred.
The older port-8083 container remains running until licensed live checks pass.
See [the presenter walkthrough](demo-valuewholesale-onprem.md).

## Isolated runtime

Run from the RedisXADK repository root:

```sh
uv sync --directory onprem
cp .env.onprem-ga.example .env.onprem-ga
chmod 600 .env.onprem-ga
```

Configure the ignored `.env.onprem-ga` before launching. `onprem/pyproject.toml` and
`onprem/uv.lock` own this runtime. The root dependency environment remains separate.
`redis-iris[fastapi-sse]==1.0.1` includes the GA clients and beta streaming integration.
`fastapi-redis-sdk==0.10.0b1` is explicitly pinned because the extra requires that prerelease.

| Settings | Required value |
|---|---|
| `REDIS_URL` | Redis for Agent Mesh streams, saved run snapshots, and member shortlists; also read by the streaming SDK. |
| `ONPREM_SESSION_SECRET` | A stable random secret of at least 32 characters, identical across workers and redeployments. |
| `IRIS_AGENT_MEMORY_URL`, `IRIS_AGENT_MEMORY_STORE_ID`, `IRIS_AGENT_MEMORY_API_KEY` | The GA data plane, a store dedicated to this demo, and a store-scoped `read,write` key. |
| `IRIS_AGENT_MEMORY_NAMESPACE_ID` | The server-issued namespace ID; a display name is insufficient. |
| `IRIS_AGENT_PLAYBOOKS_URL`, `IRIS_AGENT_PLAYBOOKS_PLAYBOOK_ID`, `IRIS_AGENT_PLAYBOOKS_API_KEY` | A playbook with published demo entries and a runtime `read` key. |
| `IRIS_INTENT_ROUTER_URL`, `IRIS_INTENT_ROUTER_ID`, `IRIS_INTENT_ROUTER_API_KEY` | The demo Router and a runtime `read` key. |
| `IRIS_CONTEXT_GATEWAY_URL`, `IRIS_CONTEXT_GATEWAY_API_KEY` | The full Gateway MCP URL and a key with the Gateway `use` grant. |
| `IRIS_LENS_ENDPOINT`, `IRIS_LENS_API_KEY`, `IRIS_LENS_AGENT_ID` | Lens ingest endpoint, ingest key, and demo agent identifier. |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_MODEL` | Vertex project, model location, and a model authorized for the runtime identity. |

Use the new deployment's actual private endpoints. Its service mapping must be verified before
reuse; the historical preview gateway mapping is not the GA mapping. The Mac needs verified
private connectivity or forwards to those endpoints; the VM needs routes to GKE and Redis.
Use Application Default Credentials locally and the VM's authorized identity in deployment.

Launch with an absolute environment-file path:

```sh
uv run --directory onprem \
  --env-file /Users/redis/code/RedisXADK/.env.onprem-ga \
  uvicorn valuewholesale_onprem.api:app --host 0.0.0.0 --port 8083 --workers 2
```

Build with the isolated directory as the context:

```sh
docker build -f onprem/Dockerfile -t valuewholesale-onprem:ga onprem
```

Service tokens, content-publishing keys, upstream credentials, and the suite license belong in
operator/deployment storage. The application environment contains scoped runtime keys only.

## Operator provisioning

First install the supported umbrella chart and configure its databases, Identity Service,
license, and inference providers. Follow the [suite installation guide](https://redis.io/docs/latest/operate/iris/self-managed/).
The verified suite package is prepared in `/Users/redis/code/Iris/.cache/ga-2026-11-04/`;
its provenance and deployment instructions are in that project’s `docs/suite-setup.md`. The
public index lacks the umbrella package, so this is explicitly a local source build with
minimal ownership-label patches, not an officially published suite archive. A valid suite
license is required before deployment.

The command shapes below were checked against `redis-iris==1.0.1`. Run them from an operator
shell with a private file containing control-plane URLs, Identity credentials, and operator
tokens. Configure `IRIS_CONFIG` and `IRIS_PROFILE` there when using a TOML profile. Replace
`/absolute/path/` placeholders and uppercase resource IDs with actual values.

```sh
iris_admin() {
  uv run --directory onprem --env-file /absolute/path/iris-operator.env \
    python -m redis_iris "$@"
}

iris_admin agent-memory service create --name valuewholesale-onprem-ga --dry-run
iris_admin agent-playbooks service create --name valuewholesale-onprem-ga --dry-run
iris_admin context-gateway service create --name valuewholesale-onprem-ga \
  --tool-exposure-mode search --dry-run
```

Create Memory and Gateway with `--write-env /absolute/path/iris-resource-admin.env` instead of
`--dry-run`. Create Playbooks and the Router using the explicit embedding options below.
Select the embedding provider/model/dimensions registered in the deployed service.
The Router's embedding choice is immutable. With the following non-secret shell variables set
to the deployment's registered values:

```sh
iris_admin agent-playbooks service create --name valuewholesale-onprem-ga \
  --embedding-provider "$IRIS_EMBEDDING_PROVIDER" \
  --embedding-model "$IRIS_EMBEDDING_MODEL" \
  --embedding-dimensions "$IRIS_EMBEDDING_DIMENSIONS" \
  --key-actions read --write-env /absolute/path/iris-resource-admin.env
iris_admin intent-router service create --name valuewholesale-onprem-ga \
  --provider "$IRIS_EMBEDDING_PROVIDER" --model "$IRIS_EMBEDDING_MODEL" \
  --dimensions "$IRIS_EMBEDDING_DIMENSIONS" --role full \
  --write-env /absolute/path/iris-resource-admin.env
```

The full Router key is for operator authoring only. Load authoring keys into the operator
process when applying content. `onprem/content/intents.json` defines the application's dispatch
references: `published_guidance`, `shopping`, and `refuse`. No match or `refuse` returns a
shopping-scope refusal. Evaluate the examples and thresholds against the deployed embedder.

Create the Memory namespace using an operator-owned store key. This verified SDK helper gets
or creates a shared namespace by name and returns its ID:

```sh
uv run --directory onprem --env-file /absolute/path/iris-resource-admin.env python - <<'END'
from redis_iris.agent_memory import AgentMemory

with AgentMemory.from_env() as memory:
    print(memory.shared_namespace("valuewholesale-onprem-ga"))
END
```

Set that ID as `IRIS_AGENT_MEMORY_NAMESPACE_ID`. The application combines owner and namespace
with `ALL` in long-term searches, and verifies both when reading a session transcript. The
namespace does not replace store grants or owner filtering.

Mint runtime keys into the private application file after recording actual resource IDs:

```sh
iris_admin agent-memory service key STORE_ID --actions read,write \
  --write /Users/redis/code/RedisXADK/.env.onprem-ga
iris_admin agent-playbooks service key PLAYBOOK_ID --actions read \
  --write /Users/redis/code/RedisXADK/.env.onprem-ga
iris_admin intent-router service key ROUTER_ID --role read \
  --write /Users/redis/code/RedisXADK/.env.onprem-ga
iris_admin context-gateway service key GATEWAY_ID \
  --write /Users/redis/code/RedisXADK/.env.onprem-ga
```

Obtain the Lens ingest key through the deployed Identity Service/console configuration. Keep a
Lens read key with the operator who inspects traces. Never deploy the operator environment.

## Approved content and business tools

The checked-in fixtures are under `onprem/content/`. Their FAQ answers preserve the fictional
policies in `data/generated/policies.jsonl` verbatim. The comparison skill is an authored
procedure. They are not automatically uploaded, published, or treated as real merchant policy.

Local schema validation requires no credentials:

```sh
uv run --directory onprem python -m redis_iris agent-playbooks faqs validate content/faqs
uv run --directory onprem python -m redis_iris agent-playbooks skills validate content/skills
```

Use a separate Playbooks content key and Router authoring key in the operator environment.
Inspect plans, then apply intentionally using absolute source paths:

```sh
iris_admin agent-playbooks service key PLAYBOOK_ID --role content \
  --write /absolute/path/iris-operator.env
iris_admin intent-router service key ROUTER_ID --role write \
  --write /absolute/path/iris-operator.env
iris_admin agent-playbooks faqs plan /Users/redis/code/RedisXADK/onprem/content/faqs
iris_admin agent-playbooks faqs sync /Users/redis/code/RedisXADK/onprem/content/faqs --publish changed
iris_admin agent-playbooks skills plan /Users/redis/code/RedisXADK/onprem/content/skills
iris_admin agent-playbooks skills sync /Users/redis/code/RedisXADK/onprem/content/skills --publish changed
iris_admin intent-router intents add --file /Users/redis/code/RedisXADK/onprem/content/intents.json
```

Only published FAQ/skill versions are served. Accepting a proposal creates an unpublished draft;
publishing is separate. The runtime cannot publish and never saves a generated answer as an
approved FAQ. Automatic proposal submission is not enabled. See
[Agent Playbooks](https://redis.io/docs/latest/develop/ai/context-engine/agent-playbooks/).

Register the approved shopping Context Surface using its actual database registry ID and data
model:

```sh
iris_admin context-gateway --gateway-id GATEWAY_ID upstream add \
  --kind context-surface --name 'Value Wholesale shopping' --namespace shopping \
  --surface-database-id SURFACE_DATABASE_ID --data-model /absolute/path/shopping-model.json
iris_admin context-gateway --gateway-id GATEWAY_ID upstream list
```

Inspect readiness and the active tool catalog; creation alone does not establish successful
discovery. Configure current catalog, inventory, member, and order records in the surface.
Disable Gateway result caching for mutable prices, inventory, orders, and balances through tool
policies. The application adds no response or business-tool cache.

The model uses `discover_business_tools`, then `call_business_tool` with the exact discovered
name/schema. Its adapter exposes read tools and binds member arguments to the selected persona.
For a generic account/order lookup without a member parameter, every returned structured record
must establish that member's ownership. Foreign, mixed-member, ownerless, and unbounded aggregate
results are rejected before entering the model. Account result handles require an explicit
schema-supported forced member constraint before creation. Expose member-specific order tools
if a generic surface omits ownership or accepts only free-form filters.
`business_result_handle` retains a large result. `handle_operation` supports advertised
page/query/count/aggregate/describe/release schemas and only handles created in the same turn.
Oversized results are rejected with paging guidance rather than silently truncated. See
[Context Gateway](https://redis.io/docs/latest/develop/ai/context-engine/context-gateway/).

## Sessions, replay, and traces

The synthetic member selector issues an HMAC-signed cookie with a random session ID; it is not
customer authentication. The cookie determines the member/session scope for requests,
snapshots, and SSE subscriptions. Use the same session secret on every worker. Set
`ONPREM_COOKIE_SECURE=true` when serving HTTPS.

`POST /api/chat` starts a scoped run. A reused run ID with different request content is rejected.
`GET /api/runs/{id}/events` replays after the cursor. A browser disconnect does not stop the
producer, and another serving worker can deliver retained events. Process failure can interrupt
the Gemini loop: Agent Mesh restores event delivery, not interrupted computation or exactly-once
external effects. Verify native stream producer-idempotency support on the actual Redis server.
[Agent Mesh documentation](https://redis.io/docs/latest/develop/ai/context-engine/agent-mesh/)
describes the beta's requirements and limits.

Snapshots at `GET /api/runs/{id}` retain the status, answer, trace ID, and bounded event history
for `ONPREM_RUN_TTL`, default `86400` seconds (24 hours). SDK stream retention is separate. The
example file sets both `REDIS_STREAM_MAX_RUN` and `REDIS_STREAM_DONE_TTL` to `3600` seconds; the
SDK itself defaults the finished-stream TTL to `120` seconds. Maximum run lifetime starts at
creation, and finishing never extends it. A saved snapshot can outlive its replayable stream.

Lens records explicit turn, memory-recall, prompt-assembly, tool, and Gemini-call spans.
`IRIS_LENS_CAPTURE_CONTENT=false` is the default. Gemini token attributes are recorded only when
returned by the model. Missing usage and costs are not estimated. SDK initialization and a trace
ID do not prove ingestion; inspect the matching exported turn in Lens. See
[Lens](https://redis.io/docs/latest/develop/ai/context-engine/lens/).

## Verification and acceptance

```sh
uv run ruff check .
uv run pytest
uv run --directory onprem ruff check .
uv run --directory onprem pytest
curl --fail http://127.0.0.1:8083/api/health
```

| Area | Local evidence | Live acceptance still required |
|---|---|---|
| Existing demos | Root lint passed and all 86 existing tests passed; separate GA lock/entrypoint. | Original container health after any deployment. |
| Content | Three FAQs and the comparison skill pass offline CLI validation. | Publish approved revisions; validate discovery with the actual embedder. |
| Memory | GA schemas and adapter tests cover owner plus namespace. | Real writes, cross-session recall, and transcript isolation. |
| Gateway | Adapter tests cover live calls, schema discovery, member binding, rejection of foreign/ownerless orders, and handle boundaries. | Actual inventory/orders, cache policies, and handle paging. |
| Mesh | Native SDK test passed against temporary local Redis 8.10.1 and two HTTP replicas: mid-run disconnect, second-replica cursor replay, one producer, completed snapshot, and denial to another member. Mock-browser recovery checks also passed. | Repeat against the actual GKE/VM network and deployed Redis, including expired-stream recovery. |
| Gemini/Lens | Local tool-loop tests pass using real GenAI types with mocked model responses; explicit spans and returned-usage handling are implemented. | A real Vertex response and the matching ingested Lens turn. |
| Container | Isolated Dockerfile and frozen lock are prepared. | Image build and container launch; the local Docker daemon was unavailable for a build. |

`/api/health` probes core services. Gemini reports client configuration; Lens reports SDK
initialization. Full inference and trace ingest remain separate checks. Record actual deployed
versions, fixture revisions, and acceptance outcomes in the shared Iris migration record.
