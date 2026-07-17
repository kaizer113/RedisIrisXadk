from __future__ import annotations

PRODUCTS = [
    {
        "sku": "VH-1001",
        "name": "Harbor Select Extra Virgin Olive Oil, 2 x 1L",
        "category": "pantry",
        "price": 24.99,
        "member_price": 21.99,
        "description": "Cold-extracted olive oil in a bulk twin pack.",
        "tags": ["cooking", "bulk", "mediterranean"],
    },
    {
        "sku": "VH-1002",
        "name": "North Trail Organic Oats, 10 lb",
        "category": "pantry",
        "price": 18.49,
        "member_price": 15.99,
        "description": "Whole-grain rolled oats for breakfast and baking.",
        "tags": ["organic", "breakfast", "bulk"],
    },
    {
        "sku": "VH-2001",
        "name": "Family Dock Paper Towels, 12 rolls",
        "category": "household",
        "price": 27.99,
        "member_price": 23.99,
        "description": "Absorbent two-ply paper towels in individually wrapped rolls.",
        "tags": ["household", "paper", "bulk"],
    },
    {
        "sku": "VH-2002",
        "name": "Clear Tide Laundry Pods, 152 count",
        "category": "household",
        "price": 36.99,
        "member_price": 31.99,
        "description": "Free-and-clear concentrated laundry detergent pods.",
        "tags": ["fragrance-free", "laundry", "bulk"],
    },
    {
        "sku": "VH-3001",
        "name": "Cascade Peak Sparkling Water, 30 x 12 oz",
        "category": "beverages",
        "price": 17.49,
        "member_price": 14.99,
        "description": "Mixed citrus, berry, and lime sparkling water.",
        "tags": ["drinks", "zero-sugar", "party"],
    },
    {
        "sku": "VH-3002",
        "name": "Rain City Medium Roast Coffee, 3 lb",
        "category": "beverages",
        "price": 29.99,
        "member_price": 25.49,
        "description": "Whole-bean medium roast with cocoa and caramel notes.",
        "tags": ["coffee", "whole-bean", "bulk"],
    },
    {
        "sku": "VH-4001",
        "name": "SummitBook 14 Laptop",
        "category": "electronics",
        "price": 899.99,
        "member_price": 849.99,
        "description": "14-inch laptop, 16 GB memory, 512 GB SSD, and two-year support.",
        "tags": ["computer", "work", "travel"],
    },
    {
        "sku": "VH-4002",
        "name": "HarborView 65-inch Mini-LED TV",
        "category": "electronics",
        "price": 1099.99,
        "member_price": 999.99,
        "description": "4K mini-LED television with a five-year member warranty.",
        "tags": ["tv", "home-theater", "warranty"],
    },
    {
        "sku": "VH-5001",
        "name": "Weeknight Salmon Portions, 3 lb",
        "category": "fresh-food",
        "price": 39.99,
        "member_price": 36.99,
        "description": "Individually portioned Atlantic salmon, ready to freeze.",
        "tags": ["seafood", "protein", "family"],
    },
    {
        "sku": "VH-5002",
        "name": "Market Garden Vegetable Tray, 4 lb",
        "category": "fresh-food",
        "price": 16.99,
        "member_price": 14.49,
        "description": "Party-size cut vegetable tray with hummus dip.",
        "tags": ["vegetarian", "party", "fresh"],
    },
]

WAREHOUSES = {
    "portland": {"name": "Portland Harbor", "city": "Portland", "state": "OR"},
    "seattle": {"name": "Seattle South", "city": "Seattle", "state": "WA"},
    "sacramento": {"name": "Sacramento River", "city": "Sacramento", "state": "CA"},
}

INVENTORY = {
    "portland": {
        "VH-1001": 42,
        "VH-1002": 31,
        "VH-2001": 18,
        "VH-2002": 0,
        "VH-3001": 66,
        "VH-3002": 23,
        "VH-4001": 4,
        "VH-4002": 2,
        "VH-5001": 15,
        "VH-5002": 8,
    },
    "seattle": {
        "VH-1001": 20,
        "VH-1002": 12,
        "VH-2001": 35,
        "VH-2002": 24,
        "VH-3001": 41,
        "VH-3002": 9,
        "VH-4001": 0,
        "VH-4002": 6,
        "VH-5001": 11,
        "VH-5002": 19,
    },
    "sacramento": {
        "VH-1001": 17,
        "VH-1002": 38,
        "VH-2001": 21,
        "VH-2002": 28,
        "VH-3001": 0,
        "VH-3002": 14,
        "VH-4001": 7,
        "VH-4002": 3,
        "VH-5001": 22,
        "VH-5002": 12,
    },
}

MEMBERS = {
    "member-1001": {
        "member_id": "member-1001",
        "name": "Alex Rivera",
        "tier": "Harbor Plus",
        "home_warehouse": "portland",
        "reward_balance": 86.42,
        "joined_at": "2022-03-18",
    },
    "member-1002": {
        "member_id": "member-1002",
        "name": "Maya Chen",
        "tier": "Harbor Standard",
        "home_warehouse": "seattle",
        "reward_balance": 21.08,
        "joined_at": "2024-09-02",
    },
    "member-1003": {
        "member_id": "member-1003",
        "name": "Jordan Brooks",
        "tier": "Harbor Plus",
        "home_warehouse": "sacramento",
        "reward_balance": 112.77,
        "joined_at": "2021-11-27",
    },
    "member-1004": {
        "member_id": "member-1004",
        "name": "Sam Patel",
        "tier": "Harbor Standard",
        "home_warehouse": "portland",
        "reward_balance": 9.15,
        "joined_at": "2025-05-14",
    },
}

ORDERS = {
    "member-1001": [
        {
            "order_id": "VH-ORD-1048",
            "placed_at": "2026-07-11",
            "status": "ready_for_pickup",
            "warehouse": "portland",
            "items": ["VH-1002", "VH-2001", "VH-3002"],
            "total": 65.47,
            "fulfillment": "warehouse_pickup",
        },
        {
            "order_id": "VH-ORD-1026",
            "placed_at": "2026-06-28",
            "status": "delivered",
            "warehouse": "portland",
            "items": ["VH-2002", "VH-3001"],
            "total": 46.98,
            "fulfillment": "delivery",
        },
    ],
    "member-1002": [
        {
            "order_id": "VH-ORD-1051",
            "placed_at": "2026-07-13",
            "status": "processing",
            "warehouse": "seattle",
            "items": ["VH-4002"],
            "total": 999.99,
            "fulfillment": "delivery",
        }
    ],
    "member-1003": [
        {
            "order_id": "VH-ORD-1042",
            "placed_at": "2026-07-06",
            "status": "delivered",
            "warehouse": "sacramento",
            "items": ["VH-1001", "VH-5001", "VH-5002"],
            "total": 73.47,
            "fulfillment": "delivery",
        },
        {
            "order_id": "VH-ORD-0998",
            "placed_at": "2026-06-09",
            "status": "picked_up",
            "warehouse": "sacramento",
            "items": ["VH-4001"],
            "total": 849.99,
            "fulfillment": "warehouse_pickup",
        },
    ],
    "member-1004": [
        {
            "order_id": "VH-ORD-1037",
            "placed_at": "2026-07-03",
            "status": "picked_up",
            "warehouse": "portland",
            "items": ["VH-1002", "VH-3002"],
            "total": 41.48,
            "fulfillment": "warehouse_pickup",
        }
    ],
}

POLICIES = [
    {
        "id": "returns",
        "title": "Member satisfaction and returns",
        "content": (
            "Most merchandise can be returned with proof of membership. Electronics have a "
            "90-day return window. Perishable goods should be reported promptly with photos "
            "when practical."
        ),
    },
    {
        "id": "pickup",
        "title": "Warehouse pickup",
        "content": (
            "Pickup orders are held for three calendar days after the ready notification. "
            "The named member must present photo identification at the pickup desk."
        ),
    },
    {
        "id": "pricing",
        "title": "Member pricing",
        "content": (
            "Displayed member prices require an active ValueHarbor membership. Prices and "
            "inventory can vary by warehouse and are confirmed when the order is submitted."
        ),
    },
]
