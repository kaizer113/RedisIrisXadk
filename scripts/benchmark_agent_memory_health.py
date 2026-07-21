"""Measure sequential Redis Agent Memory health latency with one reused SDK client."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from valuewholesale_agent.config import Settings
from valuewholesale_agent.services import MemoryService


@dataclass(frozen=True)
class Sample:
    number: int
    latency_ms: float
    error: str | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def percentile_95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


async def benchmark(count: int, timeout_ms: int) -> list[Sample]:
    settings = Settings()
    if not settings.memory_configured:
        raise RuntimeError("Redis Agent Memory settings are incomplete")

    samples: list[Sample] = []
    memory = MemoryService(settings)
    try:
        print(f"Endpoint: {settings.agent_memory_base_url}/health")
        print(
            f"Sequential requests: {count} (one reused AgentMemory client, "
            f"{settings.agent_memory_http_keepalive_seconds:g}s keep-alive)\n"
        )
        for number in range(1, count + 1):
            started = time.perf_counter_ns()
            error: str | None = None
            try:
                if memory.client is None:
                    raise RuntimeError("Redis Agent Memory client initialization failed")
                await memory.client.health_async(timeout_ms=timeout_ms)
            except Exception as exc:  # Keep sampling so failures appear in the summary.
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            samples.append(Sample(number, latency_ms, error))
            suffix = "OK" if error is None else f"ERROR {error}"
            print(f"{number:02d}: {latency_ms:8.2f} ms  {suffix}", flush=True)
    finally:
        await memory.close()
    return samples


def print_summary(samples: list[Sample]) -> None:
    successful = [sample.latency_ms for sample in samples if sample.error is None]
    failures = len(samples) - len(successful)
    print("\nSummary")
    print(f"  successful: {len(successful)}")
    print(f"  failed:     {failures}")
    if not successful:
        return
    print(f"  min:        {min(successful):8.2f} ms")
    print(f"  mean:       {statistics.fmean(successful):8.2f} ms")
    print(f"  median:     {statistics.median(successful):8.2f} ms")
    print(f"  p95:        {percentile_95(successful):8.2f} ms")
    print(f"  max:        {max(successful):8.2f} ms")
    deviation = statistics.stdev(successful) if len(successful) > 1 else 0.0
    print(f"  std dev:    {deviation:8.2f} ms")
    print(f"  first call: {successful[0]:8.2f} ms")
    if len(successful) > 1:
        print(f"  reused avg: {statistics.fmean(successful[1:]):8.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ping Redis Agent Memory health sequentially using one reusable async SDK client."
        )
    )
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--timeout-ms", type=positive_int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = asyncio.run(benchmark(args.count, args.timeout_ms))
    print_summary(samples)
    if any(sample.error is not None for sample in samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
