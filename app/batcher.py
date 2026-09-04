"""
Dynamic Batching Engine
------------------------
Collects concurrent inference requests into micro-batches to maximize
throughput on a shared "model" (here: a vectorized numpy computation
standing in for a real neural net forward pass — the batching logic
itself is model-agnostic and would work identically with a PyTorch model).

Strategy: a background asyncio task pulls requests off an asyncio.Queue.
It keeps accumulating requests until either:
  (a) MAX_BATCH_SIZE requests are collected, or
  (b) MAX_WAIT_MS has elapsed since the first request in the batch arrived
whichever comes first. This bounds both latency (via the wait timeout)
and throughput loss (via batch accumulation).
"""

import asyncio
import time
import signal
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_BATCH_SIZE = 32
MAX_WAIT_MS = 15  # max time a request waits before being force-flushed
EMBED_DIM = 768
HIDDEN_DIM = 1024
HIDDEN_DIM_2 = 512


@dataclass
class InferenceRequest:
    payload: np.ndarray
    future: asyncio.Future = field(repr=False)
    enqueued_at: float = field(default_factory=time.perf_counter)


class DummyModel:
    """Stand-in for a real neural net: a small MLP. Vectorized so batching
    actually pays off — a batch-of-N matmul is far cheaper per-sample than
    N separate matmuls, both because BLAS amortizes fixed setup/dispatch
    cost across the batch dimension and because larger matmuls use the CPU's
    cache and SIMD units more efficiently. This is real compute (no sleep()
    tricks), so the benefit shows up consistently across platforms instead
    of depending on any OS timer's sleep granularity."""

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((EMBED_DIM, HIDDEN_DIM)).astype(np.float32)
        self.W2 = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM_2)).astype(np.float32)
        self.W3 = rng.standard_normal((HIDDEN_DIM_2, 1)).astype(np.float32)

    def forward(self, batch: np.ndarray) -> np.ndarray:
        h1 = np.tanh(batch @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        out = h2 @ self.W3
        return out.squeeze(-1)


class DynamicBatcher:
    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE, max_wait_ms: float = MAX_WAIT_MS):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self.model = DummyModel()
        self._worker_task: asyncio.Task | None = None
        self._draining = False
        self._pending = 0  # requests submitted but not yet resolved; avoids a
                            # shutdown race where stop() checks an empty queue
                            # right before a concurrently-submitted request lands in it
        self.stats = {"batches": 0, "requests": 0, "batch_sizes": []}

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Graceful shutdown: stop accepting the drain flag, let the
        current in-flight batch finish, then cancel the loop."""
        self._draining = True
        if self._worker_task:
            await asyncio.wait_for(self._worker_task, timeout=2.0)

    async def submit(self, payload: np.ndarray) -> float:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending += 1
        await self.queue.put(InferenceRequest(payload=payload, future=fut))
        try:
            return await fut
        finally:
            self._pending -= 1

    async def _worker_loop(self):
        while not self._draining or self._pending > 0:
            batch = await self._collect_batch()
            if batch is None:
                continue  # no request arrived within the poll window; recheck exit condition
            loop = asyncio.get_event_loop()
            # Offload the (potentially blocking) model call to a worker
            # thread so it never blocks the event loop that's collecting
            # the *next* batch. This also sidesteps platform differences in
            # time.sleep() granularity (e.g. Windows' ~15ms default timer
            # resolution vs Linux's ~1ms) which would otherwise serialize
            # batch processing and erase the benefit of batching entirely.
            stacked = np.stack([r.payload for r in batch])
            results = await loop.run_in_executor(None, self.model.forward, stacked)
            # Futures must be resolved on the event loop thread (they aren't
            # thread-safe), so this happens back here rather than inside the
            # executor call above.
            for req, res in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(float(res))
            self.stats["batches"] += 1
            self.stats["requests"] += len(batch)
            self.stats["batch_sizes"].append(len(batch))

    async def _collect_batch(self) -> list[InferenceRequest] | None:
        try:
            first = await asyncio.wait_for(self.queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return None  # let the caller recheck the draining/exit condition
        batch = [first]
        deadline = first.enqueued_at + self.max_wait_ms / 1000

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                # The wait window is already gone — this means the worker is
                # falling behind incoming arrivals (common under load, and
                # especially likely if thread/process scheduling overhead is
                # higher, e.g. spawning executor threads on Windows). In that
                # case we should NOT just stop at whatever we've got — drain
                # everything already sitting in the queue (non-blocking) so
                # the batch grows to absorb the backlog instead of collapsing
                # to near-singleton batches under load.
                while len(batch) < self.max_batch_size:
                    try:
                        batch.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                break
            try:
                nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break
        return batch


# ---- Reference: naive one-at-a-time processor, used only by the
# benchmark script to produce the "before batching" baseline ----
class NaiveProcessor:
    def __init__(self):
        self.model = DummyModel()

    async def submit(self, payload: np.ndarray) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._infer, payload)

    def _infer(self, payload: np.ndarray) -> float:
        return float(self.model.forward(payload[None, :])[0])
