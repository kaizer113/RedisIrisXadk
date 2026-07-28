from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperienceProfile:
    id: str
    brand_name: str
    assistant_name: str
    retailer_description: str
    app_name: str
    greeting_app_name: str
    transcript_app_name: str
    agent_name: str
    greeting_agent_name: str
    dataset_relative_path: str
    static_relative_path: str
    redis_key_prefix: str
    redis_index_prefix: str
    embedding_cache_name: str
    semantic_router_index: str
    agent_memory_namespace: str
    memory_bank_display_name: str
    context_surface_name: str
    context_agent_name: str
    context_agent_display_name: str
    location_noun: str
    locations_instruction: str
    inventory_id_example: str
    sku_example: str
    categories: tuple[str, ...]
    resettable_member_ids: tuple[str, ...]
    ui_theme_stylesheet: str
    ui_favicon: str
    ui_mark: str
    ui_tagline: str
    ui_headline: str
    ui_headline_accent: str
    ui_intro: str
    ui_prompts: tuple[tuple[str, str], ...]

    @property
    def dataset_path(self) -> Path:
        return ROOT / self.dataset_relative_path

    @property
    def static_path(self) -> Path:
        return ROOT / self.static_relative_path


PROFILES = {
    "valuewholesale": ExperienceProfile(
        id="valuewholesale",
        brand_name="Value Wholesale",
        assistant_name="Vale",
        retailer_description="fictional membership warehouse retailer",
        app_name="valuewholesale-shopping-agent",
        greeting_app_name="valuewholesale-greeting-agent",
        transcript_app_name="valuewholesale-working-memory",
        agent_name="valuewholesale_shopping_agent",
        greeting_agent_name="valuewholesale_greeting_agent",
        dataset_relative_path="data/generated",
        static_relative_path="valuewholesale_agent/static",
        redis_key_prefix="valuewholesale",
        redis_index_prefix="idx:valuewholesale",
        embedding_cache_name="valuewholesale-embeddings-v1",
        semantic_router_index="valuewholesale-cache-router-v2",
        agent_memory_namespace="valuewholesale-shopping",
        memory_bank_display_name="valuewholesale-memory-bank",
        context_surface_name="Value Wholesale Shopping",
        context_agent_name="valuewholesale-adk-shopping-agent",
        context_agent_display_name="Public Value Wholesale workshop agent",
        location_noun="warehouse",
        locations_instruction=(
            "The known warehouse IDs are portland, seattle, and sacramento. "
            "Portland means the Portland Harbor warehouse (`portland`)."
        ),
        inventory_id_example="portland-vh-1001",
        sku_example="VH-1001",
        categories=("pantry", "household", "beverages", "electronics", "fresh-food"),
        resettable_member_ids=(
            "member-1001",
            "member-1002",
            "member-1003",
            "member-1004",
        ),
        ui_theme_stylesheet="/static/themes/valuewholesale.css",
        ui_favicon="/static/assets/value-wholesale-favicon.svg",
        ui_mark="VW",
        ui_tagline="Member shopping, intelligently",
        ui_headline="Stock the house.",
        ui_headline_accent="Skip the guesswork.",
        ui_intro=(
            "Meet Vale, a grounded shopping agent that understands the catalog, sees "
            "warehouse context, remembers preferences, and shows its work."
        ),
        ui_prompts=(
            ("Pantry run", "Find family-size pantry staples under $30 and check Portland stock."),
            (
                "Laundry",
                "What laundry option should I add to my order, and is it in stock in Portland?",
            ),
            ("Upcoming order", "What is in my upcoming order?"),
            ("Household products", "What household products have I bought?"),
            ("Tide Pods", "When did I last buy Tide Laundry Pods?"),
            ("Ask a policy", "What is the electronics return policy?"),
            ("Learn a product", "What flavor notes does Rain City Medium Roast Coffee have?"),
        ),
    ),
    "norlings": ExperienceProfile(
        id="norlings",
        brand_name="Norling's",
        assistant_name="Nora",
        retailer_description="fictional fashion and lifestyle department store",
        app_name="norlings-shopping-agent",
        greeting_app_name="norlings-greeting-agent",
        transcript_app_name="norlings-working-memory",
        agent_name="norlings_style_agent",
        greeting_agent_name="norlings_greeting_agent",
        dataset_relative_path="data/norlings/generated",
        static_relative_path="valuewholesale_agent/static",
        redis_key_prefix="norlings",
        redis_index_prefix="idx:norlings",
        embedding_cache_name="norlings-embeddings-v1",
        semantic_router_index="norlings-cache-router-v1",
        agent_memory_namespace="norlings-shopping",
        memory_bank_display_name="norlings-memory-bank",
        context_surface_name="Norling's Shopping",
        context_agent_name="norlings-adk-shopping-agent",
        context_agent_display_name="Public Norling's shopping agent",
        location_noun="store",
        locations_instruction=(
            "The known store IDs are manhattan, chicago, and seattle. Manhattan means "
            "Norling's Manhattan Flagship (`manhattan`), Chicago means Norling's Michigan "
            "Avenue (`chicago`), and Seattle means Norling's Downtown Seattle (`seattle`)."
        ),
        inventory_id_example="manhattan-nl-1001",
        sku_example="NL-1001",
        categories=(
            "women",
            "men",
            "shoes",
            "accessories",
            "beauty",
            "home",
            "travel",
            "electronics",
        ),
        resettable_member_ids=(
            "member-2001",
            "member-2002",
            "member-2003",
            "member-2004",
            "member-2005",
        ),
        ui_theme_stylesheet="/static/themes/norlings.css",
        ui_favicon="/static/assets/norlings-favicon.svg",
        ui_mark="NL",
        ui_tagline="Personal shopping, thoughtfully remembered",
        ui_headline="Style, considered.",
        ui_headline_accent="Chosen for you.",
        ui_intro=(
            "Meet Nora, a personal shopping agent that understands the collection, sees "
            "store availability, remembers preferences, and shows its work."
        ),
        ui_prompts=(
            ("An evening look", "Build an evening outfit under $400 and check Manhattan stock."),
            ("New-season shoes", "Which shoes fit my preferences and are available in Chicago?"),
            ("Complete the look", "What accessories would complement my recent purchases?"),
            ("Upcoming order", "What is in my upcoming order?"),
            ("Beauty edit", "Recommend a fragrance based on what you remember about me."),
            ("Ask a policy", "What is the return policy for designer items?"),
            ("Travel ready", "Find a refined carry-on option available in Seattle."),
        ),
    ),
}


def get_experience_profile(experience_id: str) -> ExperienceProfile:
    normalized = experience_id.strip().lower().replace("_", "-")
    aliases = {
        "value-wholesale": "valuewholesale",
        "look1": "valuewholesale",
        "look2": "norlings",
        "norling": "norlings",
        "norling's": "norlings",
    }
    key = aliases.get(normalized, normalized)
    try:
        return PROFILES[key]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"Unknown experience {experience_id!r}; expected one of: {available}"
        ) from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"Experience dataset is missing {path}. Generate it before starting the app."
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(frozen=True)
class ExperienceDataset:
    products: list[dict[str, Any]]
    warehouses: dict[str, dict[str, Any]]
    inventory: dict[str, dict[str, int]]
    members: dict[str, dict[str, Any]]
    orders: dict[str, list[dict[str, Any]]]
    order_items: list[dict[str, Any]]
    policies: list[dict[str, Any]]
    memory_seeds: list[dict[str, Any]]
    memory_evaluations: list[dict[str, Any]]

    @property
    def resettable_member_ids(self) -> frozenset[str]:
        return frozenset(
            str(memory["owner_id"])
            for memory in self.memory_seeds
            if memory.get("owner_id") in self.members
        )


@lru_cache
def load_experience_dataset(dataset_dir: str) -> ExperienceDataset:
    root = Path(dataset_dir)
    products = _read_jsonl(root / "products.jsonl")
    warehouse_records = _read_jsonl(root / "warehouses.jsonl")
    inventory_records = _read_jsonl(root / "inventory.jsonl")
    member_records = _read_jsonl(root / "members.jsonl")
    order_records = _read_jsonl(root / "orders.jsonl")
    order_items = _read_jsonl(root / "order_items.jsonl")

    warehouses = {
        str(item["warehouse_id"]): {
            key: value for key, value in item.items() if key != "warehouse_id"
        }
        for item in warehouse_records
    }
    inventory: dict[str, dict[str, int]] = {}
    for item in inventory_records:
        inventory.setdefault(str(item["warehouse_id"]), {})[str(item["sku"])] = int(
            item["quantity"]
        )
    members = {str(item["member_id"]): item for item in member_records}

    items_by_order: dict[str, list[dict[str, Any]]] = {}
    for item in order_items:
        items_by_order.setdefault(str(item["order_id"]), []).append(item)
    orders: dict[str, list[dict[str, Any]]] = {}
    for item in order_records:
        normalized = dict(item)
        lines = sorted(
            items_by_order.get(str(item["order_id"]), []),
            key=lambda row: row["line_number"],
        )
        normalized["items"] = [
            str(line["sku"]) for line in lines for _ in range(max(1, int(line.get("quantity", 1))))
        ]
        orders.setdefault(str(item["member_id"]), []).append(normalized)

    return ExperienceDataset(
        products=products,
        warehouses=warehouses,
        inventory=inventory,
        members=members,
        orders=orders,
        order_items=order_items,
        policies=_read_jsonl(root / "policies.jsonl"),
        memory_seeds=_read_jsonl(root / "memory_seeds.jsonl"),
        memory_evaluations=_read_jsonl(root / "memory_evaluations.jsonl"),
    )
