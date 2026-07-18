from __future__ import annotations

import argparse
import asyncio

from valuewholesale_agent.config import get_settings
from valuewholesale_agent.services import LangCacheService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset mutable state that affects the repeatable Value Wholesale demo flow."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive LangCache flush.",
    )
    return parser.parse_args()


async def reset_demo(confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit("Refusing to flush LangCache without --yes")

    settings = get_settings()
    cache = LangCacheService(settings)
    if not cache.base_url:
        raise SystemExit("LangCache is not configured")

    await cache.clear()
    print("LangCache flushed. Reload the browser to start a fresh application session.")


if __name__ == "__main__":
    asyncio.run(reset_demo(parse_args().yes))
