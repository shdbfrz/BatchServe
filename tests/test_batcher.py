import asyncio
import numpy as np
import pytest

from app.batcher import DynamicBatcher, NaiveProcessor, EMBED_DIM


@pytest.mark.asyncio
async def test_single_request_returns_result():
    batcher = DynamicBatcher()
    await batcher.start()
    payload = np.zeros(EMBED_DIM, dtype=np.float32)
    result = await batcher.submit(payload)
    assert isinstance(result, float)
    await batcher.stop()


@pytest.mark.asyncio
async def test_concurrent_requests_are_batched_together():
    batcher = DynamicBatcher(max_batch_size=16, max_wait_ms=50)
    await batcher.start()

    async def one():
        payload = np.random.default_rng().standard_normal(EMBED_DIM).astype(np.float32)
        return await batcher.submit(payload)

    results = await asyncio.gather(*(one() for _ in range(20)))
    assert len(results) == 20
    assert all(isinstance(r, float) for r in results)
    # 20 requests with batch cap 16 must span at least 2 batches
    assert batcher.stats["batches"] >= 2
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_respects_max_batch_size():
    batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)
    await batcher.start()

    async def one():
        payload = np.zeros(EMBED_DIM, dtype=np.float32)
        return await batcher.submit(payload)

    await asyncio.gather(*(one() for _ in range(12)))
    assert max(batcher.stats["batch_sizes"]) <= 4
    await batcher.stop()


@pytest.mark.asyncio
async def test_naive_processor_matches_batched_determinism():
    # Same model weights (seeded) -> naive and batched paths should agree
    # on a zero vector (deterministic input).
    naive = NaiveProcessor()
    batched = DynamicBatcher()
    await batched.start()

    zero = np.zeros(EMBED_DIM, dtype=np.float32)
    r_naive = await naive.submit(zero)
    r_batched = await batched.submit(zero)

    assert abs(r_naive - r_batched) < 1e-4
    await batched.stop()


@pytest.mark.asyncio
async def test_graceful_shutdown_drains_queue():
    batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=20)
    await batcher.start()

    async def one():
        payload = np.zeros(EMBED_DIM, dtype=np.float32)
        return await batcher.submit(payload)

    task = asyncio.gather(*(one() for _ in range(30)))
    await asyncio.sleep(0.01)  # let requests actually reach the queue
    await batcher.stop()  # should drain remaining queued items before exit
    results = await task
    assert len(results) == 30
