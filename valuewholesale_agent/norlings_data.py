from __future__ import annotations

from typing import Any

PRODUCTS = [
    {
        "sku": "NL-1001",
        "name": "Norling's Signature Wool Wrap Coat",
        "category": "women",
        "price": 298.00,
        "member_price": 268.00,
        "description": "Double-face wool-blend wrap coat with a removable belt.",
        "tags": ["coat", "wool", "outerwear", "workwear"],
    },
    {
        "sku": "NL-1002",
        "name": "Elodie Silk Midi Dress",
        "category": "women",
        "price": 189.00,
        "member_price": 169.00,
        "description": "Bias-cut silk midi dress with a softly draped neckline.",
        "tags": ["dress", "silk", "occasion", "midi"],
    },
    {
        "sku": "NL-1003",
        "name": "Arden Cashmere Cardigan",
        "category": "women",
        "price": 149.00,
        "member_price": 129.00,
        "description": "Lightweight cashmere cardigan with pearlized buttons.",
        "tags": ["cashmere", "cardigan", "layering"],
    },
    {
        "sku": "NL-1004",
        "name": "Mercer Tailored Trousers",
        "category": "women",
        "price": 119.00,
        "member_price": 99.00,
        "description": "High-rise stretch-wool trousers with a straight leg.",
        "tags": ["trousers", "workwear", "tailored"],
    },
    {
        "sku": "NL-2001",
        "name": "Norling's Modern Merino Blazer",
        "category": "men",
        "price": 249.00,
        "member_price": 219.00,
        "description": "Unstructured merino-blend blazer for travel and work.",
        "tags": ["blazer", "merino", "tailored", "travel"],
    },
    {
        "sku": "NL-2002",
        "name": "Sutton Supima Oxford Shirt",
        "category": "men",
        "price": 89.00,
        "member_price": 74.00,
        "description": "Trim-fit Supima cotton oxford shirt with an easy-care finish.",
        "tags": ["shirt", "cotton", "workwear"],
    },
    {
        "sku": "NL-2003",
        "name": "Westbridge Leather Sneaker",
        "category": "men",
        "price": 139.00,
        "member_price": 119.00,
        "description": "Minimal leather sneaker with a cushioned recycled-rubber sole.",
        "tags": ["sneaker", "leather", "casual"],
    },
    {
        "sku": "NL-3001",
        "name": "Fifth Avenue Chelsea Boot",
        "category": "shoes",
        "price": 179.00,
        "member_price": 159.00,
        "description": "Water-resistant leather Chelsea boot with a stacked heel.",
        "tags": ["boots", "leather", "weather-resistant"],
    },
    {
        "sku": "NL-3002",
        "name": "Gallery Italian Leather Tote",
        "category": "accessories",
        "price": 229.00,
        "member_price": 199.00,
        "description": "Structured Italian leather tote sized for a 14-inch laptop.",
        "tags": ["tote", "leather", "work", "travel"],
    },
    {
        "sku": "NL-3003",
        "name": "Jardin Printed Silk Scarf",
        "category": "accessories",
        "price": 79.00,
        "member_price": 69.00,
        "description": "Hand-rolled silk scarf with an abstract botanical print.",
        "tags": ["scarf", "silk", "gift"],
    },
    {
        "sku": "NL-4001",
        "name": "Lumière Renewal Face Serum",
        "category": "beauty",
        "price": 68.00,
        "member_price": 62.00,
        "description": "Fragrance-free hydrating serum with niacinamide and peptides.",
        "tags": ["skincare", "serum", "fragrance-free"],
    },
    {
        "sku": "NL-4002",
        "name": "Maison Norling No. 7 Eau de Parfum",
        "category": "beauty",
        "price": 95.00,
        "member_price": 85.00,
        "description": "Warm amber fragrance with bergamot, iris, and sandalwood.",
        "tags": ["fragrance", "amber", "gift"],
    },
    {
        "sku": "NL-4003",
        "name": "Atelier Satin Lip Color — Rosewood",
        "category": "beauty",
        "price": 34.00,
        "member_price": 30.00,
        "description": "Satin lipstick in a neutral rosewood shade.",
        "tags": ["makeup", "lipstick", "rosewood"],
    },
    {
        "sku": "NL-5001",
        "name": "Hotel Collection Percale Duvet Set",
        "category": "home",
        "price": 159.00,
        "member_price": 139.00,
        "description": "Crisp 400-thread-count cotton percale duvet cover and shams.",
        "tags": ["bedding", "cotton", "percale"],
    },
    {
        "sku": "NL-5002",
        "name": "Spa Weight Turkish Cotton Towels, Set of 6",
        "category": "home",
        "price": 72.00,
        "member_price": 64.00,
        "description": "Plush long-staple cotton bath towel set.",
        "tags": ["towels", "cotton", "bath"],
    },
    {
        "sku": "NL-5003",
        "name": "Cedar & Fig Ceramic Candle",
        "category": "home",
        "price": 42.00,
        "member_price": 38.00,
        "description": "Hand-poured soy candle in a reusable ceramic vessel.",
        "tags": ["candle", "home-fragrance", "gift"],
    },
    {
        "sku": "NL-6001",
        "name": "Avenue Carry-On Spinner",
        "category": "travel",
        "price": 245.00,
        "member_price": 219.00,
        "description": "Lightweight polycarbonate carry-on with compression panels.",
        "tags": ["luggage", "carry-on", "travel"],
    },
    {
        "sku": "NL-6002",
        "name": "Studio Wireless Noise-Canceling Headphones",
        "category": "electronics",
        "price": 199.00,
        "member_price": 179.00,
        "description": "Over-ear wireless headphones with adaptive noise cancellation.",
        "tags": ["headphones", "wireless", "travel"],
    },
]

WAREHOUSES = {
    "manhattan": {"name": "Norling's Manhattan Flagship", "city": "New York", "state": "NY"},
    "chicago": {"name": "Norling's Michigan Avenue", "city": "Chicago", "state": "IL"},
    "seattle": {"name": "Norling's Downtown Seattle", "city": "Seattle", "state": "WA"},
}

_QUANTITIES = {
    "manhattan": [8, 12, 15, 10, 7, 18, 11, 6, 9, 22, 20, 14, 25, 10, 16, 24, 5, 9],
    "chicago": [5, 8, 9, 14, 10, 13, 16, 9, 4, 18, 17, 11, 21, 7, 12, 15, 3, 6],
    "seattle": [0, 6, 12, 8, 5, 10, 14, 13, 7, 16, 19, 8, 18, 9, 14, 20, 6, 4],
}
INVENTORY = {
    store_id: {
        product["sku"]: quantity for product, quantity in zip(PRODUCTS, quantities, strict=True)
    }
    for store_id, quantities in _QUANTITIES.items()
}

MEMBERS = {
    "member-2001": {
        "member_id": "member-2001",
        "name": "Ava Thompson",
        "tier": "Norling's Icon",
        "home_warehouse": "manhattan",
        "reward_balance": 184.50,
        "joined_at": "2021-09-14",
    },
    "member-2002": {
        "member_id": "member-2002",
        "name": "Marcus Lee",
        "tier": "Norling's Insider",
        "home_warehouse": "chicago",
        "reward_balance": 42.00,
        "joined_at": "2024-02-08",
    },
    "member-2003": {
        "member_id": "member-2003",
        "name": "Priya Shah",
        "tier": "Norling's Icon",
        "home_warehouse": "seattle",
        "reward_balance": 96.75,
        "joined_at": "2022-05-21",
    },
    "member-2004": {
        "member_id": "member-2004",
        "name": "Elena Garcia",
        "tier": "Norling's Insider",
        "home_warehouse": "manhattan",
        "reward_balance": 18.25,
        "joined_at": "2025-01-12",
    },
    "member-2005": {
        "member_id": "member-2005",
        "name": "Daniel Kim",
        "tier": "Norling's Icon",
        "home_warehouse": "chicago",
        "reward_balance": 131.40,
        "joined_at": "2020-11-03",
    },
}

ORDERS = {
    "member-2001": [
        {
            "order_id": "NL-ORD-2001",
            "placed_at": "2026-07-18",
            "status": "delivered",
            "warehouse": "manhattan",
            "items": ["NL-1001", "NL-3003"],
            "total": 337.00,
            "fulfillment": "delivery",
        },
        {
            "order_id": "NL-ORD-2002",
            "placed_at": "2026-07-25",
            "status": "ready_for_pickup",
            "warehouse": "manhattan",
            "items": ["NL-4003"],
            "total": 30.00,
            "fulfillment": "warehouse_pickup",
        },
    ],
    "member-2002": [
        {
            "order_id": "NL-ORD-2003",
            "placed_at": "2026-07-23",
            "status": "processing",
            "warehouse": "chicago",
            "items": ["NL-2001", "NL-2002"],
            "total": 293.00,
            "fulfillment": "delivery",
        }
    ],
    "member-2003": [
        {
            "order_id": "NL-ORD-2004",
            "placed_at": "2026-07-09",
            "status": "delivered",
            "warehouse": "seattle",
            "items": ["NL-4001", "NL-4002"],
            "total": 147.00,
            "fulfillment": "delivery",
        }
    ],
    "member-2004": [
        {
            "order_id": "NL-ORD-2005",
            "placed_at": "2026-07-26",
            "status": "ready_for_pickup",
            "warehouse": "manhattan",
            "items": ["NL-5001", "NL-5002", "NL-5002"],
            "total": 267.00,
            "fulfillment": "warehouse_pickup",
        }
    ],
    "member-2005": [
        {
            "order_id": "NL-ORD-2006",
            "placed_at": "2026-06-30",
            "status": "picked_up",
            "warehouse": "chicago",
            "items": ["NL-2003", "NL-6002"],
            "total": 298.00,
            "fulfillment": "warehouse_pickup",
        }
    ],
}

POLICIES = [
    {
        "id": "returns",
        "title": "Returns and exchanges",
        "content": (
            "Most unworn merchandise with original tags may be returned within 30 days. "
            "Norling's Icon members receive 45 days. Final-sale merchandise is ineligible, "
            "and approved refunds return to the original form of payment."
        ),
    },
    {
        "id": "beauty-returns",
        "title": "Beauty returns",
        "content": (
            "Unopened beauty products may be returned within 30 days. Opened products are "
            "accepted only when defective or associated with an adverse reaction."
        ),
    },
    {
        "id": "shipping-pickup",
        "title": "Shipping and store pickup",
        "content": (
            "Standard shipping is complimentary on orders of 89 dollars or more. Store pickup "
            "orders are held for five calendar days and require member identification."
        ),
    },
    {
        "id": "pricing-rewards",
        "title": "Norling's pricing and rewards",
        "content": (
            "Norling's prices require an active account. Rewards accrue on eligible merchandise "
            "and exclude taxes, gift cards, and services."
        ),
    },
    {
        "id": "alterations",
        "title": "Alterations",
        "content": (
            "Basic alterations are complimentary on full-price tailored apparel for Icon members. "
            "Timing and complex alterations are quoted by the selected store."
        ),
    },
]

MEMORY_SEEDS = [
    {
        "id": "mem-2001-style",
        "owner_id": "member-2001",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Ava prefers neutral-colored tailored workwear in wool and cashmere.",
        "topics": ["shopping", "women", "workwear", "preference"],
    },
    {
        "id": "mem-2001-pickup",
        "owner_id": "member-2001",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Ava prefers pickup at the Norling's Manhattan Flagship.",
        "topics": ["shopping", "fulfillment", "preference"],
    },
    {
        "id": "mem-2002-fit",
        "owner_id": "member-2002",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Marcus usually buys trim-fit men's shirts and size 40 regular blazers.",
        "topics": ["shopping", "men", "fit", "preference"],
    },
    {
        "id": "mem-2002-event",
        "owner_id": "member-2002",
        "namespace": "norlings-shopping",
        "memory_type": "episodic",
        "text": "Marcus compared leather sneakers for an August city trip but did not purchase.",
        "topics": ["shopping", "men", "travel", "browsing"],
    },
    {
        "id": "mem-2003-beauty",
        "owner_id": "member-2003",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Priya prefers fragrance-free skincare with niacinamide.",
        "topics": ["shopping", "beauty", "preference"],
    },
    {
        "id": "mem-2003-gift",
        "owner_id": "member-2003",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Priya often chooses warm amber fragrances as gifts.",
        "topics": ["shopping", "beauty", "gift"],
    },
    {
        "id": "mem-2004-home",
        "owner_id": "member-2004",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Elena prefers crisp cotton percale bedding in white or ivory.",
        "topics": ["shopping", "home", "bedding", "preference"],
    },
    {
        "id": "mem-2004-pickup",
        "owner_id": "member-2004",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Elena prefers pickup at the Manhattan Flagship after work.",
        "topics": ["shopping", "fulfillment", "schedule"],
    },
    {
        "id": "mem-2005-travel",
        "owner_id": "member-2005",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Daniel prefers lightweight carry-on luggage and noise-canceling headphones.",
        "topics": ["shopping", "travel", "electronics", "preference"],
    },
    {
        "id": "mem-2005-shoes",
        "owner_id": "member-2005",
        "namespace": "norlings-shopping",
        "memory_type": "semantic",
        "text": "Daniel prefers minimal leather sneakers in neutral colors.",
        "topics": ["shopping", "men", "shoes", "preference"],
    },
]

MEMORY_EVALUATIONS = [
    {
        "case_id": "norlings-memory-001",
        "member_id": "member-2001",
        "query": "Recommend workwear and the best fulfillment option for me.",
        "expected_terms": ["tailored", "wool", "Manhattan", "pickup"],
        "relevant_memory_ids": ["mem-2001-style", "mem-2001-pickup"],
    },
    {
        "case_id": "norlings-memory-002",
        "member_id": "member-2002",
        "query": "What fit and travel footwear suit me?",
        "expected_terms": ["trim-fit", "40 regular", "leather sneakers"],
        "relevant_memory_ids": ["mem-2002-fit", "mem-2002-event"],
    },
    {
        "case_id": "norlings-memory-003",
        "member_id": "member-2003",
        "query": "Recommend skincare and a gift fragrance.",
        "expected_terms": ["fragrance-free", "niacinamide", "amber"],
        "relevant_memory_ids": ["mem-2003-beauty", "mem-2003-gift"],
    },
    {
        "case_id": "norlings-memory-004",
        "member_id": "member-2004",
        "query": "Recommend bedding and pickup timing.",
        "expected_terms": ["percale", "ivory", "Manhattan", "after work"],
        "relevant_memory_ids": ["mem-2004-home", "mem-2004-pickup"],
    },
    {
        "case_id": "norlings-memory-005",
        "member_id": "member-2005",
        "query": "What should I pack for quiet travel?",
        "expected_terms": ["carry-on", "noise-canceling", "leather sneakers"],
        "relevant_memory_ids": ["mem-2005-travel", "mem-2005-shoes"],
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
            "updated_at": "2026-07-27T16:00:00Z",
        }
        for warehouse_id, stock in INVENTORY.items()
        for sku, quantity in stock.items()
    ]
    members = [dict(member) for member in MEMBERS.values()]
    product_by_sku = {product["sku"]: product for product in PRODUCTS}
    orders: list[dict[str, Any]] = []
    order_items: list[dict[str, Any]] = []
    for member_id, member_orders in ORDERS.items():
        for order in member_orders:
            normalized = {key: value for key, value in order.items() if key != "items"}
            normalized["member_id"] = member_id
            normalized["item_count"] = len(order["items"])
            orders.append(normalized)
            quantities: dict[str, int] = {}
            for sku in order["items"]:
                quantities[sku] = quantities.get(sku, 0) + 1
            for line_number, (sku, quantity) in enumerate(quantities.items(), start=1):
                product = product_by_sku[sku]
                order_items.append(
                    {
                        "order_item_id": f"{order['order_id']}-{line_number}",
                        "order_id": order["order_id"],
                        "line_number": line_number,
                        "sku": sku,
                        "product_name": product["name"],
                        "quantity": quantity,
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
