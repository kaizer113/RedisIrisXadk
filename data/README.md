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
| `memory_seeds.jsonl` | `id` | Synthetic memory records |
| `memory_evaluations.jsonl` | `case_id` | Synthetic retrieval checks |

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

The keys are lowercase and colon-separated. Product, policy, member, order, and order-item prefixes
are indexed by Redis Query Engine. Inventory remains a direct O(1) lookup after product discovery.
