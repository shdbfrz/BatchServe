# BatchServe — Dynamic Batching Inference Server

An asynchronous inference gateway that opportunistically groups concurrent
prediction requests into micro-batches before running them through the model,
trading a few milliseconds of queueing latency for a large throughput gain.
Built with FastAPI + asyncio, instrumented with Prometheus, containerized
with Docker, and provisioned with a Grafana dashboard via Compose.

## Why batching helps

A neural net forward pass on a batch of N inputs is usually much cheaper
than N separate forward passes on 1 input each — most of the cost (kernel
launch overhead, memory movement, Python/IPC overhead) is per *call*, not
per *sample*. `DynamicBatcher` exploits this by holding incoming requests in
an `asyncio.Queue` and flushing a batch when either:
- `MAX_BATCH_SIZE` (32) requests have accumulated, or
- `MAX_WAIT_MS` (15ms) has elapsed since the oldest queued request arrived

whichever happens first — bounding both added latency and wasted throughput.

## Results (measured on this machine, not simulated)

Ran via `benchmark.py`, which exercises the real `DynamicBatcher` class
directly — 500 requests, concurrency 50:

| Mode              | Throughput   |
|-------------------|--------------|
| Naive (1-at-a-time) | ~3,200–4,100 req/s |
| Dynamic batching    | ~8,800–9,500 req/s |

**~130–200% throughput improvement**, average batch size 31.2/32.

The model does real vectorized compute (a 3-layer MLP, no artificial
`sleep()` calls standing in for "work") so the gain comes from genuine BLAS
batch amortization — larger matmuls use the CPU's cache/SIMD more
efficiently and spread fixed per-call dispatch cost across more samples.
This also makes the benchmark's outcome consistent across platforms, since
it no longer depends on any OS's sleep-timer granularity (see the dev-notes
below for why that mattered).

Re-run it yourself:
```bash
pip install -r requirements.txt
PYTHONPATH=. python3 benchmark.py
```

> Note: the "model" is a small 3-layer MLP (real matrix multiplications, no
> artificial delays) standing in for a real network, so the batching engine
> can be exercised without a GPU. The batching logic is model-agnostic —
> swap the model call in `app/batcher.py` for a real PyTorch/ONNX model and
> the queueing/flushing behavior is unchanged.

### Dev notes: two bugs this design surfaced (kept here for anyone extending this)

1. **Never put a blocking call directly in the async worker loop.** An
   earlier version used `time.sleep()` to simulate per-call model overhead,
   called directly inside the async batch-processing coroutine. On Linux
   this was nearly invisible (sleep granularity ~1ms), but on Windows,
   where `time.sleep()`'s effective resolution is much coarser (commonly
   ~15ms), it silently serialized every batch and made batching *slower*
   than the naive baseline. Fix: offload blocking model calls via
   `loop.run_in_executor(...)`, and resolve the futures back on the event
   loop thread afterward (asyncio Futures aren't thread-safe).
2. **When the worker falls behind, batches should grow, not collapse.** The
   batch-collection loop originally broke out as soon as its wait-deadline
   passed, even if more requests were already sitting in the queue. Under
   backlog (exactly the situation thread-dispatch overhead can cause), this
   produced near-singleton batches — the opposite of what dynamic batching
   should do under load. Fix: once the deadline passes, drain whatever's
   already queued (non-blocking) instead of stopping immediately.
3. Finally, the sleep-based overhead simulation itself was replaced with
   real (if small) matrix multiplications, so the benchmark's outcome
   reflects actual compute amortization rather than a platform-dependent
   timer trick.

## Features

- **Async dynamic batching** — `asyncio.Queue` + background worker loop,
  size- and time-bounded flushing (`app/batcher.py`)
- **Graceful shutdown** — SIGTERM handler drains in-flight requests before
  exit, using a pending-request counter (not just queue emptiness) to avoid
  a shutdown race where a request could be dropped mid-submit
- **Prometheus metrics** — request count, batch count, latency histogram,
  batch-size histogram exposed at `/metrics`
- **Grafana dashboard provisioning** — auto-connected via Docker Compose
- **Docker** — `--timeout-graceful-shutdown` gives the drain logic headroom
  before the container runtime sends SIGKILL

## Running it

**Locally:**
```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```

**With Docker Compose** (app + Prometheus + Grafana):
```bash
docker compose up --build
# app:        http://localhost:8000
# prometheus: http://localhost:9090
# grafana:    http://localhost:3000
```

## API

| Endpoint         | Description                                  |
|------------------|-----------------------------------------------|
| `GET /`          | Live telemetry dashboard (fire requests, watch real-time batching) |
| `POST /predict`  | Submit a vector (or omit for a random one); returns score + latency |
| `GET /stats`     | Running totals: requests, batches, avg/max batch size, recent batch-size history |
| `GET /metrics`   | Prometheus exposition format |
| `GET /health`    | Liveness check |
| `GET /docs`      | Auto-generated Swagger UI |

## Testing

```bash
PYTHONPATH=. pytest tests/ -v
```
5 tests covering: single-request correctness, concurrent batching behavior,
max-batch-size enforcement, naive-vs-batched output parity, and graceful
shutdown draining. All passing.

## Project structure

```
BatchServe/
├── app/
│   ├── batcher.py   # DynamicBatcher engine + DummyModel + NaiveProcessor baseline
│   └── main.py      # FastAPI app, lifespan/SIGTERM handling, Prometheus metrics
├── tests/
│   └── test_batcher.py
├── benchmark.py     # naive vs. batched throughput comparison
├── Dockerfile
├── docker-compose.yml
├── monitoring/      # Prometheus + Grafana provisioning
└── requirements.txt
```
