from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from valueharbor_agent.demo_data import (
    INVENTORY,
    MEMBERS,
    ORDERS,
    POLICIES,
    PRODUCTS,
    WAREHOUSES,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "generated"

MEMORY_SEEDS = [
    {
        "id": "mem-1001-household",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex prefers fragrance-free household and laundry products.",
        "topics": ["shopping", "household", "preference"],
    },
    {
        "id": "mem-1001-pickup",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex prefers warehouse pickup at the Portland Harbor location.",
        "topics": ["shopping", "fulfillment", "preference"],
    },
    {
        "id": "mem-1001-shopping-time",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex usually shops on Saturday mornings before 10 AM.",
        "topics": ["shopping", "schedule", "preference"],
    },
    {
        "id": "mem-1001-receipts",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex prefers receipts delivered by email instead of printed copies.",
        "topics": ["shopping", "receipts", "preference"],
    },
    {
        "id": "mem-1001-snacks",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex likes lightly salted bulk snacks for neighborhood gatherings.",
        "topics": ["shopping", "food", "preference"],
    },
    {
        "id": "mem-1001-telescope",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "episodic",
        "text": "Alex compared two beginner telescopes last winter but did not purchase one.",
        "topics": ["shopping", "outdoors", "browsing"],
    },
    {
        "id": "mem-1001-tire-event",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "episodic",
        "text": "Alex attended a tire safety event at the warehouse in May 2026.",
        "topics": ["warehouse", "automotive", "event"],
    },
    {
        "id": "mem-1001-patio-return",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "episodic",
        "text": "Alex returned a patio umbrella because it was too large for the balcony.",
        "topics": ["shopping", "outdoors", "return"],
    },
    {
        "id": "mem-1001-book-club",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Alex hosts a mystery book club on the first Thursday of each month.",
        "topics": ["personal", "books", "schedule"],
    },
    {
        "id": "mem-1001-tote",
        "owner_id": "member-1001",
        "namespace": "valueharbor-shopping",
        "memory_type": "episodic",
        "text": "Alex brought a blue insulated reusable tote on the last warehouse visit.",
        "topics": ["warehouse", "visit", "accessory"],
    },
    {
        "id": "mem-1002-theater",
        "owner_id": "member-1002",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Maya is interested in home theater products and extended warranties.",
        "topics": ["shopping", "electronics", "interest"],
    },
    {
        "id": "mem-1002-delivery",
        "owner_id": "member-1002",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Maya prefers home delivery from the Seattle South warehouse.",
        "topics": ["shopping", "fulfillment", "preference"],
    },
    {
        "id": "mem-1003-food",
        "owner_id": "member-1003",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Jordan often shops for organic pantry staples and seafood protein.",
        "topics": ["shopping", "food", "interest"],
    },
    {
        "id": "mem-1003-laptop",
        "owner_id": "member-1003",
        "namespace": "valueharbor-shopping",
        "memory_type": "episodic",
        "text": "Jordan purchased a SummitBook laptop for travel in June 2026.",
        "topics": ["shopping", "electronics", "purchase"],
    },
    {
        "id": "mem-1004-coffee",
        "owner_id": "member-1004",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Sam prefers whole-bean medium roast coffee.",
        "topics": ["shopping", "beverages", "preference"],
    },
    {
        "id": "mem-1004-food",
        "owner_id": "member-1004",
        "namespace": "valueharbor-shopping",
        "memory_type": "semantic",
        "text": "Sam prefers vegetarian party food and avoids seafood recommendations.",
        "topics": ["shopping", "food", "preference"],
    },
]

MEMORY_EVALUATIONS = [
    {
        "case_id": "memory-eval-001",
        "member_id": "member-1001",
        "query": "What laundry products and pickup options should you recommend for me?",
        "expected_terms": ["fragrance-free", "Portland", "pickup"],
        "relevant_memory_ids": ["mem-1001-household", "mem-1001-pickup"],
    },
    {
        "case_id": "memory-eval-002",
        "member_id": "member-1002",
        "query": "How should I receive a new television and what else interests me?",
        "expected_terms": ["delivery", "home theater", "warranties"],
        "relevant_memory_ids": ["mem-1002-theater", "mem-1002-delivery"],
    },
    {
        "case_id": "memory-eval-003",
        "member_id": "member-1003",
        "query": "Recommend groceries that fit my usual shopping interests.",
        "expected_terms": ["organic", "seafood", "protein"],
        "relevant_memory_ids": ["mem-1003-food"],
    },
    {
        "case_id": "memory-eval-004",
        "member_id": "member-1003",
        "query": "What computer did I recently buy and why?",
        "expected_terms": ["SummitBook", "travel", "June 2026"],
        "relevant_memory_ids": ["mem-1003-laptop"],
    },
    {
        "case_id": "memory-eval-005",
        "member_id": "member-1004",
        "query": "What beverages and party foods match my preferences?",
        "expected_terms": ["whole-bean", "medium roast", "vegetarian"],
        "relevant_memory_ids": ["mem-1004-coffee", "mem-1004-food"],
    },
    {
        "case_id": "memory-eval-006",
        "member_id": "member-1001",
        "query": "Which laundry detergent fits my preferences?",
        "expected_terms": ["fragrance-free", "household", "laundry"],
        "relevant_memory_ids": ["mem-1001-household"],
        "distractor_memory_ids": [
            "mem-1001-telescope",
            "mem-1001-tire-event",
            "mem-1001-patio-return",
            "mem-1001-book-club",
            "mem-1001-tote",
        ],
    },
    {
        "case_id": "memory-eval-007",
        "member_id": "member-1001",
        "query": "Where should I collect my next order?",
        "expected_terms": ["Portland", "warehouse", "pickup"],
        "relevant_memory_ids": ["mem-1001-pickup"],
        "distractor_memory_ids": [
            "mem-1001-shopping-time",
            "mem-1001-receipts",
            "mem-1001-snacks",
            "mem-1001-book-club",
        ],
    },
]


def records() -> dict[str, list[dict[str, Any]]]:
    products = [dict(product) for product in PRODUCTS]
    warehouses = [
        {"warehouse_id": warehouse_id, **warehouse}
        for warehouse_id, warehouse in WAREHOUSES.items()
    ]
    inventory = [
        {
            "inventory_id": f"{warehouse_id}-{sku.lower()}",
            "warehouse_id": warehouse_id,
            "sku": sku,
            "quantity": quantity,
            "updated_at": "2026-07-16T16:00:00Z",
        }
        for warehouse_id, stock in INVENTORY.items()
        for sku, quantity in stock.items()
    ]
    members = [dict(member) for member in MEMBERS.values()]

    orders: list[dict[str, Any]] = []
    order_items: list[dict[str, Any]] = []
    product_by_sku = {product["sku"]: product for product in PRODUCTS}
    for member_id, member_orders in ORDERS.items():
        for order in member_orders:
            normalized_order = {key: value for key, value in order.items() if key != "items"}
            normalized_order["member_id"] = member_id
            normalized_order["item_count"] = len(order["items"])
            orders.append(normalized_order)
            for line_number, sku in enumerate(order["items"], start=1):
                product = product_by_sku[sku]
                order_items.append(
                    {
                        "order_item_id": f"{order['order_id']}-{line_number}",
                        "order_id": order["order_id"],
                        "line_number": line_number,
                        "sku": sku,
                        "product_name": product["name"],
                        "quantity": 1,
                        "unit_price": product["member_price"],
                    }
                )

    return {
        "products": products,
        "warehouses": warehouses,
        "inventory": inventory,
        "members": members,
        "orders": orders,
        "order_items": order_items,
        "policies": [dict(policy) for policy in POLICIES],
        "memory_seeds": [dict(memory) for memory in MEMORY_SEEDS],
        "memory_evaluations": [dict(case) for case in MEMORY_EVALUATIONS],
    }


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = records()
    for name, items in datasets.items():
        write_jsonl(output_dir / f"{name}.jsonl", items)

    manifest = {
        "name": "valueharbor-demo",
        "version": "1.0.0",
        "generated_at": "2026-07-16T16:00:00Z",
        "format": "jsonl",
        "entities": {name: len(items) for name, items in datasets.items()},
        "relationships": [
            "inventory.warehouse_id -> warehouses.warehouse_id",
            "inventory.sku -> products.sku",
            "orders.member_id -> members.member_id",
            "orders.warehouse -> warehouses.warehouse_id",
            "order_items.order_id -> orders.order_id",
            "order_items.sku -> products.sku",
            "memory_seeds.owner_id -> members.member_id",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ValueHarbor demo dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate(args.output)
    print(json.dumps(manifest["entities"], sort_keys=True))


if __name__ == "__main__":
    main()
