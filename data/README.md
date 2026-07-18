# Value Wholesale demo dataset

This is a deterministic, fictional commerce dataset for the shopping-agent workshop. Run `make dataset` to regenerate `data/generated`.

## Entities

| File | Primary identifier | Purpose |
|---|---|---|
| `products.jsonl` | `sku` | Searchable product catalog and pricing |
| `warehouses.jsonl` | `warehouse_id` | Warehouse locations |
| `inventory.jsonl` | `inventory_id` | Per-warehouse product availability |
| `members.jsonl` | `member_id` | Fictional signed-in customer profiles |
| `orders.jsonl` | `order_id` | Order headers and fulfillment state |
| `order_items.jsonl` | `order_item_id` | Normalized order lines |
| `policies.jsonl` | `id` | Grounding documents for policy RAG |
| `memory_seeds.jsonl` | `id` | Identical facts for seeding both memory systems |
| `memory_evaluations.jsonl` | `case_id` | Queries and labeled relevance expectations |

All names, orders, prices, and preferences are synthetic. No real customer data is included.

## Redis model

The seed loader uses flat Hashes for independently searchable entities and Strings for atomic inventory quantities:

```text
valuewholesale:product:{sku}                         Hash
valuewholesale:warehouse:{warehouse_id}              Hash
valuewholesale:inventory:{warehouse_id}:{sku}        String integer
valuewholesale:member:{member_id}                    Hash
valuewholesale:order:{order_id}                      Hash
valuewholesale:order-item:{order_item_id}             Hash
valuewholesale:policy:{policy_id}                    Hash
valuewholesale:memory-seed:{memory_id}                Hash staging record
valuewholesale:memory-evaluation:{case_id}            Hash staging record
```

The keys are lowercase and colon-separated. Product, policy, member, order, and order-item prefixes are indexed by Redis Query Engine. Inventory remains a direct O(1) lookup because the agent normally knows both the warehouse and SKU after product discovery. Memory seed records in the database are the reproducible source corpus; they will also be copied into each managed memory service for comparison.

## Memory comparison methodology

`memory_seeds.jsonl` provides the same facts for Redis Agent Memory and Vertex ADK Memory Bank. Each evaluation case includes:

- one member-scoped retrieval query;
- expected terms used to calculate transparent precision@k and recall@k;
- the IDs of the memories considered relevant.
- optional IDs of deliberately irrelevant or weakly related memories used as distractors.

The `member-1001` corpus includes durable preferences plus realistic but query-irrelevant
episodic facts. Both providers receive the identical corpus. This makes it possible to show the
precision cost of top-k-only retrieval when a provider cannot apply a similarity threshold.

`member-1005` (Taylor Morgan) is the high-cardinality test user with exactly 500 memories:
20 durable semantic shopping preferences and 480 dated episodic browsing interactions. This
corpus exercises owner-scoped retrieval latency and relevance under a much noisier history.

For a fair comparison, seed both systems from the same file, wait for asynchronous indexing or promotion, warm each provider once, then report medians over multiple measured runs. The corpus is identical, while each provider retains its native retrieval controls: Redis Agent Memory applies its configured similarity threshold and ADK Memory Bank returns its top-k matches.
