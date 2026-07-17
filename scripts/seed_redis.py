from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from valueharbor_agent.services import CatalogService, services

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"


def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Dataset file is missing: {path}. Run `make dataset` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def redis_mapping(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in record.items()
        if value is not None
    }


def ensure_indexes(catalog: CatalogService) -> None:
    client = catalog.redis
    if client is None:
        raise SystemExit("REDIS_URL is required")
    indexes = {
        item.decode() if isinstance(item, bytes) else str(item)
        for item in client.execute_command("FT._LIST")
    }
    if "idx:valueharbor:products" not in indexes:
        client.execute_command(
            "FT.CREATE",
            "idx:valueharbor:products",
            "ON",
            "HASH",
            "PREFIX",
            1,
            "valueharbor:product:",
            "SCHEMA",
            "sku",
            "TAG",
            "name",
            "TEXT",
            "WEIGHT",
            2,
            "category",
            "TAG",
            "price",
            "NUMERIC",
            "SORTABLE",
            "member_price",
            "NUMERIC",
            "SORTABLE",
            "description",
            "TEXT",
            "tags",
            "TAG",
            "embedding",
            "VECTOR",
            "HNSW",
            10,
            "TYPE",
            "FLOAT32",
            "DIM",
            768,
            "DISTANCE_METRIC",
            "COSINE",
            "M",
            16,
            "EF_CONSTRUCTION",
            200,
        )
    if "idx:valueharbor:policies" not in indexes:
        client.execute_command(
            "FT.CREATE",
            "idx:valueharbor:policies",
            "ON",
            "HASH",
            "PREFIX",
            1,
            "valueharbor:policy:",
            "SCHEMA",
            "title",
            "TEXT",
            "WEIGHT",
            2,
            "content",
            "TEXT",
        )
    if "idx:valueharbor:members" not in indexes:
        client.execute_command(
            "FT.CREATE",
            "idx:valueharbor:members",
            "ON",
            "HASH",
            "PREFIX",
            1,
            "valueharbor:member:",
            "SCHEMA",
            "member_id",
            "TAG",
            "name",
            "TEXT",
            "tier",
            "TAG",
            "home_warehouse",
            "TAG",
            "reward_balance",
            "NUMERIC",
        )
    if "idx:valueharbor:orders" not in indexes:
        client.execute_command(
            "FT.CREATE",
            "idx:valueharbor:orders",
            "ON",
            "HASH",
            "PREFIX",
            1,
            "valueharbor:order:",
            "SCHEMA",
            "order_id",
            "TAG",
            "member_id",
            "TAG",
            "status",
            "TAG",
            "warehouse",
            "TAG",
            "fulfillment",
            "TAG",
            "placed_at",
            "TAG",
            "total",
            "NUMERIC",
        )
    if "idx:valueharbor:order-items" not in indexes:
        client.execute_command(
            "FT.CREATE",
            "idx:valueharbor:order-items",
            "ON",
            "HASH",
            "PREFIX",
            1,
            "valueharbor:order-item:",
            "SCHEMA",
            "order_item_id",
            "TAG",
            "order_id",
            "TAG",
            "sku",
            "TAG",
            "product_name",
            "TEXT",
            "quantity",
            "NUMERIC",
            "unit_price",
            "NUMERIC",
        )


def main() -> None:
    catalog = services.catalog
    client = catalog.redis
    if client is None:
        raise SystemExit("REDIS_URL is required")
    ensure_indexes(catalog)

    products = load_jsonl("products")
    warehouses = load_jsonl("warehouses")
    inventory = load_jsonl("inventory")
    members = load_jsonl("members")
    orders = load_jsonl("orders")
    order_items = load_jsonl("order_items")
    policies = load_jsonl("policies")
    memory_seeds = load_jsonl("memory_seeds")
    memory_evaluations = load_jsonl("memory_evaluations")

    pipeline = client.pipeline(transaction=False)
    for product in products:
        embedding = catalog._embed(f"{product['name']}. {product['description']}")  # noqa: SLF001
        mapping = redis_mapping(product)
        mapping["tags"] = ",".join(product["tags"])
        if embedding:
            mapping["embedding"] = embedding
        key = f"valueharbor:product:{product['sku']}"
        pipeline.delete(key)
        pipeline.hset(key, mapping=mapping)
    for warehouse in warehouses:
        key = f"valueharbor:warehouse:{warehouse['warehouse_id']}"
        pipeline.delete(key)
        pipeline.hset(
            key,
            mapping=redis_mapping(warehouse),
        )
    for stock in inventory:
        pipeline.set(
            f"valueharbor:inventory:{stock['warehouse_id']}:{stock['sku']}",
            stock["quantity"],
        )
    for member in members:
        key = f"valueharbor:member:{member['member_id']}"
        pipeline.delete(key)
        pipeline.hset(key, mapping=redis_mapping(member))
    for order in orders:
        key = f"valueharbor:order:{order['order_id']}"
        pipeline.delete(key)
        pipeline.hset(key, mapping=redis_mapping(order))
    for item in order_items:
        key = f"valueharbor:order-item:{item['order_item_id']}"
        pipeline.delete(key)
        pipeline.hset(
            key, mapping=redis_mapping(item)
        )
    for policy in policies:
        key = f"valueharbor:policy:{policy['id']}"
        pipeline.delete(key)
        pipeline.hset(key, mapping=redis_mapping(policy))
    for memory in memory_seeds:
        pipeline.hset(f"valueharbor:memory-seed:{memory['id']}", mapping=redis_mapping(memory))
    for evaluation in memory_evaluations:
        pipeline.hset(
            f"valueharbor:memory-evaluation:{evaluation['case_id']}",
            mapping=redis_mapping(evaluation),
        )
    pipeline.execute()
    print(
        "Seeded "
        f"{len(products)} products, {len(warehouses)} warehouses, {len(inventory)} stock records, "
        f"{len(members)} members, {len(orders)} orders, {len(order_items)} order items, "
        f"{len(policies)} policies, {len(memory_seeds)} memory seeds, "
        f"and {len(memory_evaluations)} memory evaluations."
    )


if __name__ == "__main__":
    main()
