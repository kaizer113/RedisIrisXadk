# Self-managed Iris Value Wholesale demo

This third demo is an exact Value Wholesale application copy whose Redis Iris calls go to the
self-managed GKE deployment. The original Value Wholesale and Norling's containers continue to
use their Redis Cloud services.

## Runtime isolation

| Concern | Third-demo value |
|---|---|
| Environment file | `.env.onprem` (local and ignored) |
| Example configuration | `.env.onprem.example` |
| VM container | `valuewholesale-onprem-agent` |
| Public port | `8083` |
| Redis key prefix | `valuewholesale-onprem` |
| Redis index prefix | `idx:valuewholesale-onprem` |
| Embedding cache | `valuewholesale-onprem-embeddings-v1` |
| Semantic router index | `valuewholesale-onprem-cache-router-v1` |
| Agent Memory namespace | `valuewholesale-onprem-shopping` |
| Context Surface | `Value Wholesale Shopping On-Prem` |

The catalog Redis database is shared safely because every application key, search index,
embedding cache, tool-call cache, semantic-router index, and Context Retriever record uses the
third-demo namespace. Do not use database-wide destructive commands such as `FLUSHDB`: Redis also
holds metadata and data for the self-managed Iris services.

Google ADK, Vertex sessions, Gemini models, and ADK Memory Bank remain configured exactly as in
the original Value Wholesale `.env`.

## Private service endpoints

The saved runtime configuration uses the GKE private gateway on `10.42.0.9`:

- Agent Memory data plane: port `9000`
- LangCache data plane: port `9001`
- Context Retriever admin: port `8080` (setup only)
- Context Retriever MCP: port `8081`, path `/mcp`

Credentials and resource identifiers come from `/Users/redis/code/Iris/.secrets/` and are never
committed. Agent Memory authentication is disabled on this private demo endpoint; the app uses a
non-secret placeholder API-key value because its adapter requires all three configuration fields
to be non-empty.

## Setup from the Mac

The private endpoints are reachable from the VM, not directly from the Mac. Open SSH forwards and
override only the setup process URLs; keep the saved `.env.onprem` URLs VM-private:

```bash
gcloud compute ssh valuewholesale-demo --zone us-east4-c -- \
  -N \
  -L 19000:10.42.0.9:9000 \
  -L 19001:10.42.0.9:9001 \
  -L 18080:10.42.0.9:8080 \
  -L 18081:10.42.0.9:8081

CTX_API_URL=http://127.0.0.1:18080 \
CTX_MCP_URL=http://127.0.0.1:18081/mcp \
make setup-context EXPERIENCE=valuewholesale ENV_FILE=.env.onprem

AGENT_MEMORY_BASE_URL=http://127.0.0.1:19000 \
EXPERIENCE_ID=valuewholesale \
uv run --env-file .env.onprem python -m scripts.seed_managed_memories \
  --env-file .env.onprem --provider redis
```

Use `make seed EXPERIENCE=valuewholesale ENV_FILE=.env.onprem` for the isolated catalog/index
data. ADK Memory Bank does not need another seed because this copy intentionally retains the
original ADK configuration.

## Compatibility findings

Verified on September 28, 2026:

- Agent Memory SDK `0.2.0` works with the self-managed `0.7.0` data plane for health, session
  writes/reads, bulk long-term writes, filtered vector search, inventory listing, and deletion.
- LangCache uses the same entry/search/flush API as Redis Cloud, but self-managed deployments
  additionally require a Control Plane cache and an Identity Service agent key scoped to that
  cache. Set/search/delete and the application cache-aside flow passed.
- Context Retriever's MCP URL must include `/mcp`. The setup tool now builds an explicit `ctxctl`
  profile when `CTX_API_URL` or `CTX_MCP_URL` is supplied and accepts the on-prem empty-list
  response.
- Before semantic search was enabled, the Context Surface generated 36 governed tools, compared
  with 52 from the Redis Cloud surface.
  The cloud service exposes field-specific names such as `filter_order_by_member_id`; self-managed
  `0.4.2` instead exposes generic tools such as `filter_order` plus `union_results`,
  `intersect_results`, `except_results`, and `expand_results`. This is not tool-name parity, but
  ADK discovery adapted successfully and member, order, and order-item retrieval passed against
  the isolated keys.
- The self-managed LangCache uses Vertex `text-embedding-005` at 768 dimensions. This is separate
  from the application's local 384-dimensional RedisVL embedding model and does not conflict.
- Context Retriever semantic search is enabled through the internal TLS endpoint
  `https://vertex-embeddings-tls/v1`, backed by Vertex `text-embedding-005` at 768 dimensions.
  The on-prem Product model links `semantic_embedding` to `description`. This adds
  `search_product_by_semantic_embedding_semantic` and
  `search_product_by_semantic_embedding_hybrid`, bringing the surface to 38 tools. Direct MCP
  tests passed for unfiltered semantic search, hybrid search, and category-filtered semantic
  search. The application still exposes its separate RedisVL catalog tool, so Gemini may choose
  that existing tool for ordinary catalog prompts.
- The two-worker container took longer than the old 60-second deployment readiness window while
  both workers loaded the local embedding model. The VM deploy script now allows two minutes.
- Agent Memory and Context Retriever licenses expire November 27, 2026. LangCache's license also
  expires November 27, 2026. Renew them before relying on the demo after that date.

The private Agent Memory endpoint has authentication disabled and the gateway uses plain HTTP.
That is acceptable only for this restricted demo network; it is not a production security model.
