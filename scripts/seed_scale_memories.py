"""Optionally seed a larger hidden-user corpus into both long-term memory providers."""

from __future__ import annotations

import argparse
import json
from typing import Any

from scripts.seed_managed_memories import seed_redis, seed_vertex
from valuewholesale_agent.config import Settings

DEFAULT_USERS = 100
DEFAULT_MEMORIES_PER_USER = 100
ADK_WRITES_PER_MINUTE = 80

PREFERENCE_TEMPLATES = [
    ("Prefers fragrance-free household products.", ["household", "preference"]),
    ("Prefers warehouse pickup over home delivery.", ["fulfillment", "preference"]),
    ("Usually shops on weekend mornings.", ["schedule", "preference"]),
    ("Prefers email receipts instead of printed copies.", ["receipts", "preference"]),
    ("Likes lightly salted snacks for gatherings.", ["food", "preference"]),
    ("Prefers whole-bean medium-roast coffee.", ["beverages", "preference"]),
    ("Looks for organic pantry staples.", ["pantry", "preference"]),
    ("Prefers vegetarian party food.", ["food", "preference"]),
    ("Buys paper goods in bulk packages.", ["household", "preference"]),
    ("Prefers unscented laundry detergent.", ["laundry", "preference"]),
    ("Looks for electronics with extended warranties.", ["electronics", "preference"]),
    ("Prefers curbside collection when available.", ["fulfillment", "preference"]),
    ("Usually compares member price before choosing.", ["pricing", "preference"]),
    ("Prefers recyclable product packaging.", ["sustainability", "preference"]),
    ("Often buys shelf-stable food for meal planning.", ["pantry", "preference"]),
    ("Prefers decaffeinated coffee after noon.", ["beverages", "preference"]),
    ("Looks for low-sodium pantry options.", ["food", "preference"]),
    ("Prefers compact packages for limited storage.", ["storage", "preference"]),
    ("Usually checks local warehouse stock before visiting.", ["inventory", "preference"]),
    ("Prefers multipacks with individually wrapped portions.", ["packaging", "preference"]),
]

EVENT_PRODUCTS = [
    "rolled oats",
    "olive oil",
    "laundry detergent",
    "paper towels",
    "whole-bean coffee",
    "sparkling water",
    "pasta",
    "tomato sauce",
    "storage bins",
    "batteries",
    "laptop computers",
    "wireless speakers",
    "cookware",
    "folding tables",
    "cooler bags",
    "water filters",
    "office chairs",
    "garden supplies",
    "breakfast cereal",
    "reusable food containers",
]

EVENT_ACTIONS = [
    ("Browsed {product} and saved the item for later.", ["browsing", "saved-item"]),
    ("Compared several {product} options during a warehouse visit.", ["comparison"]),
    ("Asked about member pricing for {product}.", ["pricing", "question"]),
    ("Checked warehouse availability for {product}.", ["inventory", "availability"]),
]


def memory_templates() -> list[dict[str, Any]]:
    templates = [
        {"text": text, "memory_type": "semantic", "topics": topics}
        for text, topics in PREFERENCE_TEMPLATES
    ]
    templates.extend(
        {
            "text": pattern.format(product=product),
            "memory_type": "episodic",
            "topics": ["shopping", *topics],
        }
        for product in EVENT_PRODUCTS
        for pattern, topics in EVENT_ACTIONS
    )
    if len(templates) != DEFAULT_MEMORIES_PER_USER:
        raise RuntimeError(f"Expected 100 reusable memory templates, found {len(templates)}")
    return templates


def build_memories(
    *,
    namespace: str,
    start_user: int = 1,
    users: int = DEFAULT_USERS,
    memories_per_user: int = DEFAULT_MEMORIES_PER_USER,
) -> list[dict[str, Any]]:
    if start_user < 1:
        raise ValueError("start_user must be at least 1")
    if users < 1:
        raise ValueError("users must be at least 1")
    if not 1 <= memories_per_user <= DEFAULT_MEMORIES_PER_USER:
        raise ValueError("memories_per_user must be between 1 and 100")
    templates = memory_templates()[:memories_per_user]
    memories = []
    for user_number in range(start_user, start_user + users):
        owner_id = f"scale-member-{user_number:04d}"
        for memory_number, template in enumerate(templates, start=1):
            memories.append(
                {
                    "id": f"scale-{user_number:04d}-{memory_number:03d}",
                    "owner_id": owner_id,
                    "namespace": namespace,
                    **template,
                }
            )
    return memories


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a reusable 100-memory corpus for hidden synthetic owners into Redis Agent "
            "Memory and ADK Memory Bank."
        )
    )
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--memories-per-user", type=int, default=DEFAULT_MEMORIES_PER_USER)
    parser.add_argument("--start-user", type=int, default=1)
    parser.add_argument(
        "--provider",
        choices=("both", "redis", "vertex"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the external writes; required unless --dry-run is used.",
    )
    args = parser.parse_args()
    if not args.dry_run and not args.yes:
        parser.error("--yes is required because this command writes managed memory records")

    settings = Settings()
    memories = build_memories(
        namespace=settings.agent_memory_namespace,
        start_user=args.start_user,
        users=args.users,
        memories_per_user=args.memories_per_user,
    )
    estimate_minutes = len(memories) / ADK_WRITES_PER_MINUTE
    summary = {
        "users": args.users,
        "memories_per_user": args.memories_per_user,
        "total_memories": len(memories),
        "first_owner_id": memories[0]["owner_id"],
        "last_owner_id": memories[-1]["owner_id"],
        "namespace": settings.agent_memory_namespace,
        "provider": args.provider,
        "estimated_adk_minutes": round(estimate_minutes, 1)
        if args.provider in {"both", "vertex"}
        else 0,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return

    if args.provider in {"both", "redis"}:
        created, errors = seed_redis(settings, memories)
        print(f"Redis Agent Memory: {created} created, {errors} errors", flush=True)
        if errors:
            raise SystemExit(1)
    if args.provider in {"both", "vertex"}:
        created, skipped = seed_vertex(settings, memories)
        print(
            f"ADK Memory Bank: {created} created, {skipped} already present",
            flush=True,
        )


if __name__ == "__main__":
    main()
