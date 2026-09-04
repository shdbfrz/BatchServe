"""
Benchmark: naive one-request-at-a-time processing vs. DynamicBatcher.
Run directly (no server needed) — exercises the same code paths the
FastAPI app uses, so the numbers reflect the real engine.
"""
import asyncio
import time
import numpy as np

from app.batcher import DynamicBatcher, NaiveProcessor, EMBED_DIM

N_REQUESTS = 2000
CONCURRENCY = 200


async def run_naive():
    proc = NaiveProcessor()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one():
        async with sem:
            payload = np.random.default_rng().standard_normal(EMBED_DIM).astype(np.float32)
            await proc.submit(payload)

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(N_REQUESTS)))
    return time.perf_counter() - start


async def run_batched():
    batcher = DynamicBatcher()
    await batcher.start()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one():
        async with sem:
            payload = np.random.default_rng().standard_normal(EMBED_DIM).astype(np.float32)
            await batcher.submit(payload)

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(N_REQUESTS)))
    elapsed = time.perf_counter() - start
    await batcher.stop()
    return elapsed, batcher.stats


async def main():
    print(f"Benchmarking with {N_REQUESTS} requests, concurrency={CONCURRENCY}\n")

    naive_time = await run_naive()
    naive_throughput = N_REQUESTS / naive_time
    print(f"[Naive]   total={naive_time:.3f}s  throughput={naive_throughput:.1f} req/s")

    batched_time, stats = await run_batched()
    batched_throughput = N_REQUESTS / batched_time
    avg_batch = sum(stats["batch_sizes"]) / len(stats["batch_sizes"])
    print(f"[Batched] total={batched_time:.3f}s  throughput={batched_throughput:.1f} req/s "
          f"  batches={stats['batches']}  avg_batch_size={avg_batch:.1f}")

    improvement = (batched_throughput / naive_throughput - 1) * 100
    print(f"\n>>> Throughput improvement: {improvement:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
