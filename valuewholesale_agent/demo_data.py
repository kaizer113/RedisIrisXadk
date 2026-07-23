from __future__ import annotations

CORE_PRODUCTS = [
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
        "sku": "VH-1011",
        "name": "Rain City 72% Dark Chocolate Squares, 24 count",
        "category": "pantry",
        "price": 21.99,
        "member_price": 18.99,
        "description": "Individually wrapped dark chocolate squares with a rich cocoa finish.",
        "tags": ["chocolate", "dark-chocolate", "dessert", "individually-wrapped"],
    },
    {
        "sku": "VH-1012",
        "name": "Cascade Creamy Milk Chocolate Bars, 18 count",
        "category": "pantry",
        "price": 19.49,
        "member_price": 16.99,
        "description": "Smooth milk chocolate bars in a shareable club-size box.",
        "tags": ["chocolate", "milk-chocolate", "dessert", "shareable"],
    },
    {
        "sku": "VH-1013",
        "name": "Harbor Select Sea Salt Caramels, 32 oz",
        "category": "pantry",
        "price": 23.99,
        "member_price": 20.49,
        "description": "Dark chocolate-covered soft caramels finished with flaky sea salt.",
        "tags": ["chocolate", "caramel", "sea-salt", "gift"],
    },
    {
        "sku": "VH-1014",
        "name": "North Fork Chocolate Hazelnut Truffles, 48 count",
        "category": "pantry",
        "price": 27.99,
        "member_price": 24.49,
        "description": "Milk chocolate truffles with creamy roasted hazelnut centers.",
        "tags": ["chocolate", "truffles", "hazelnut", "gift"],
    },
    {
        "sku": "VH-1015",
        "name": "Orchard Trail Dark Chocolate Almond Clusters, 36 oz",
        "category": "pantry",
        "price": 22.49,
        "member_price": 19.49,
        "description": "Roasted almond clusters coated in dark chocolate for a crunchy snack.",
        "tags": ["chocolate", "dark-chocolate", "almonds", "snack"],
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

# Thirty product families with three deterministic pack variants produce 90 additional products.
# Keeping this source compact and generated makes the 105-product workshop catalog reproducible.
_CATALOG_FAMILIES = [
    (
        "pantry",
        "Harbor Trail Lightly Salted Tortilla Chips",
        "Crunchy corn tortilla chips with a light sea-salt finish.",
        ["lightly-salted", "snack", "party"],
        8.99,
    ),
    (
        "pantry",
        "North Fork Roasted Almonds",
        "Dry-roasted almonds seasoned with a small amount of sea salt.",
        ["nuts", "protein", "lightly-salted"],
        13.49,
    ),
    (
        "pantry",
        "Portside Bronze-Cut Pasta",
        "Slow-dried bronze-cut pasta for family meals and gatherings.",
        ["pasta", "italian", "bulk"],
        9.49,
    ),
    (
        "pantry",
        "Willamette Jasmine Rice",
        "Aromatic long-grain jasmine rice for everyday cooking.",
        ["rice", "gluten-free", "bulk"],
        14.99,
    ),
    (
        "pantry",
        "Canyon Creek Black Beans",
        "Low-sodium black beans packed for soups, bowls, and sides.",
        ["beans", "low-sodium", "pantry"],
        10.49,
    ),
    (
        "pantry",
        "Rainier Honey Oat Granola",
        "Toasted oat clusters with honey, seeds, and dried fruit.",
        ["breakfast", "snack", "whole-grain"],
        11.99,
    ),
    (
        "household",
        "Clear Harbor Dishwasher Tablets",
        "Unscented concentrated dishwasher tablets for everyday loads.",
        ["unscented", "dishwasher", "cleaning"],
        17.99,
    ),
    (
        "household",
        "Family Dock Drawstring Trash Bags",
        "Tear-resistant drawstring bags for kitchen and household use.",
        ["trash-bags", "durable", "bulk"],
        19.99,
    ),
    (
        "household",
        "Pure Current Foaming Hand Soap",
        "Fragrance-free foaming hand soap in refillable containers.",
        ["fragrance-free", "soap", "refill"],
        12.49,
    ),
    (
        "household",
        "Harbor Scrub Non-Scratch Sponges",
        "Reusable non-scratch sponges for dishes and counters.",
        ["sponges", "kitchen", "reusable"],
        8.49,
    ),
    (
        "household",
        "Soft Landing Facial Tissues",
        "Three-ply facial tissues in family-size multipacks.",
        ["tissues", "paper", "bulk"],
        14.49,
    ),
    (
        "household",
        "Northline Recycled Aluminum Foil",
        "Heavy-duty recycled aluminum foil for cooking and storage.",
        ["foil", "kitchen", "recycled"],
        15.99,
    ),
    (
        "beverages",
        "Cascade Meadow Herbal Tea",
        "Caffeine-free herbal tea with mint and citrus botanicals.",
        ["tea", "caffeine-free", "variety"],
        12.99,
    ),
    (
        "beverages",
        "Orchard Press Apple Juice",
        "Not-from-concentrate apple juice with no added sugar.",
        ["juice", "no-added-sugar", "family"],
        13.49,
    ),
    (
        "beverages",
        "Summit Hydration Sports Drink",
        "Electrolyte drink variety pack with reduced sugar.",
        ["sports-drink", "electrolytes", "variety"],
        16.99,
    ),
    (
        "beverages",
        "Rain City Dark Cocoa Mix",
        "Rich cocoa drink mix made with Dutch-process cocoa.",
        ["cocoa", "hot-drink", "bulk"],
        10.99,
    ),
    (
        "beverages",
        "Portside Cold Brew Coffee",
        "Smooth ready-to-drink cold brew coffee concentrate.",
        ["coffee", "cold-brew", "concentrate"],
        18.99,
    ),
    (
        "beverages",
        "Canyon Grove Sparkling Lemonade",
        "Lightly sweetened sparkling lemonade in assorted citrus flavors.",
        ["lemonade", "sparkling", "party"],
        15.49,
    ),
    (
        "electronics",
        "Northstar Wireless Earbuds",
        "Noise-isolating wireless earbuds with a charging case.",
        ["audio", "wireless", "travel"],
        49.99,
    ),
    (
        "electronics",
        "TrailCharge USB-C Power Bank",
        "Portable fast-charging battery with two USB-C ports.",
        ["charging", "portable", "travel"],
        39.99,
    ),
    (
        "electronics",
        "HarborView 27-inch Monitor",
        "QHD monitor with an adjustable stand and USB-C input.",
        ["monitor", "office", "usb-c"],
        229.99,
    ),
    (
        "electronics",
        "SummitKeys Wireless Keyboard",
        "Low-profile wireless keyboard with multi-device pairing.",
        ["keyboard", "office", "wireless"],
        44.99,
    ),
    (
        "electronics",
        "RainSound Portable Speaker",
        "Water-resistant Bluetooth speaker with all-day battery life.",
        ["speaker", "bluetooth", "outdoor"],
        59.99,
    ),
    (
        "electronics",
        "PortLink Mesh Wi-Fi Router",
        "Dual-band mesh router system for whole-home coverage.",
        ["networking", "wifi", "home"],
        179.99,
    ),
    (
        "fresh-food",
        "Cedar Grove Chicken Breasts",
        "Individually packed boneless chicken breasts ready to freeze.",
        ["chicken", "protein", "family"],
        24.99,
    ),
    (
        "fresh-food",
        "Columbia Valley Honeycrisp Apples",
        "Crisp sweet-tart apples grown in the Pacific Northwest.",
        ["fruit", "apples", "snack"],
        12.99,
    ),
    (
        "fresh-food",
        "Tillamook Country Cheese Slices",
        "Mild cheddar and Colby Jack slices for lunches and gatherings.",
        ["cheese", "snack", "party"],
        14.99,
    ),
    (
        "fresh-food",
        "Market Garden Classic Hummus",
        "Creamy chickpea hummus made with tahini and lemon.",
        ["hummus", "vegetarian", "snack"],
        9.99,
    ),
    (
        "fresh-food",
        "North Meadow Greek Yogurt",
        "Plain high-protein Greek yogurt in family-size tubs.",
        ["yogurt", "protein", "breakfast"],
        11.49,
    ),
    (
        "fresh-food",
        "Cascade Berry Medley",
        "Fresh strawberries, blueberries, and raspberries for sharing.",
        ["berries", "fruit", "fresh"],
        13.99,
    ),
]

_PACK_VARIANTS = [
    ("Value Pack", 1.0),
    ("Family Pack", 1.35),
    ("Club Pack", 1.75),
]


def _expanded_products() -> list[dict]:
    products = []
    for family_index, (category, name, description, tags, base_price) in enumerate(
        _CATALOG_FAMILIES
    ):
        for variant_index, (pack_name, multiplier) in enumerate(_PACK_VARIANTS):
            member_price = round(base_price * multiplier, 2)
            products.append(
                {
                    "sku": f"VH-{6001 + family_index * 3 + variant_index}",
                    "name": f"{name}, {pack_name}",
                    "category": category,
                    "price": round(member_price * 1.14, 2),
                    "member_price": member_price,
                    "description": f"{description.rstrip('.')} {pack_name.lower()}.",
                    "tags": [*tags, "value-wholesale", pack_name.lower().replace(" ", "-")],
                }
            )
    return products


PRODUCTS = [*CORE_PRODUCTS, *_expanded_products()]

WAREHOUSES = {
    "portland": {"name": "Portland Harbor", "city": "Portland", "state": "OR"},
    "seattle": {"name": "Seattle South", "city": "Seattle", "state": "WA"},
    "sacramento": {"name": "Sacramento River", "city": "Sacramento", "state": "CA"},
}

_CORE_INVENTORY = {
    "portland": {
        "VH-1001": 42,
        "VH-1002": 31,
        "VH-1011": 34,
        "VH-1012": 28,
        "VH-1013": 21,
        "VH-1014": 16,
        "VH-1015": 25,
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
        "VH-1011": 27,
        "VH-1012": 32,
        "VH-1013": 18,
        "VH-1014": 23,
        "VH-1015": 19,
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
        "VH-1011": 22,
        "VH-1012": 26,
        "VH-1013": 29,
        "VH-1014": 14,
        "VH-1015": 31,
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


def _expanded_inventory() -> dict[str, dict[str, int]]:
    inventory = {warehouse_id: dict(stock) for warehouse_id, stock in _CORE_INVENTORY.items()}
    for warehouse_index, warehouse_id in enumerate(inventory):
        for product_index, product in enumerate(PRODUCTS[len(CORE_PRODUCTS) :]):
            quantity = (product_index * 17 + warehouse_index * 23 + 11) % 72
            inventory[warehouse_id][product["sku"]] = 0 if quantity % 19 == 0 else quantity
    return inventory


INVENTORY = _expanded_inventory()

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
    "member-1005": {
        "member_id": "member-1005",
        "name": "Taylor Morgan",
        "tier": "Harbor Plus",
        "home_warehouse": "seattle",
        "reward_balance": 64.31,
        "joined_at": "2020-08-22",
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
            "Displayed member prices require an active Value Wholesale membership. Prices and "
            "inventory can vary by warehouse and are confirmed when the order is submitted."
        ),
    },
]
