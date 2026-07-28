# Value Wholesale demo dataset

This is a deterministic, fictional commerce dataset for the shopping-agent workshop. Run `make dataset` to regenerate `data/generated`.

## Experience datasets

The default Value Wholesale records live in `data/generated`. The second Norling's experience lives
in `data/norlings/generated` and intentionally stays small:

- 5 customers;
- 3 stores;
- 18 products and 54 store-inventory records;
- 6 orders and 11 normalized order lines;
- 5 policies;
- 10 long-term memory seeds, exactly 2 per customer;
- 5 memory retrieval evaluation cases.

Generate either dataset with `make dataset EXPERIENCE=valuewholesale` or
`make dataset EXPERIENCE=norlings`. Runtime fallback data, Redis seeding, Context Retriever imports,
and managed-memory seeding all use the selected experience's generated JSONL directory.

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
