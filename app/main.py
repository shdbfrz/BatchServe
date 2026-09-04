import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.batcher import DynamicBatcher, EMBED_DIM

REQUEST_COUNT = Counter("batchserve_requests_total", "Total inference requests")
BATCH_COUNT = Counter("batchserve_batches_total", "Total batches processed")
LATENCY = Histogram("batchserve_request_latency_seconds", "End-to-end request latency")
BATCH_SIZE_HIST = Histogram(
    "batchserve_batch_size", "Distribution of batch sizes",
    buckets=(1, 2, 4, 8, 16, 32, 64)
)

batcher = DynamicBatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await batcher.start()

    loop = __import__("asyncio").get_event_loop()

    def _sigterm_handler():
        print("[BatchServe] SIGTERM received, draining in-flight batch before shutdown...")
        loop.create_task(batcher.stop())

    try:
        loop.add_signal_handler(signal.SIGTERM, _sigterm_handler)
    except NotImplementedError:
        pass  # not supported on some platforms (e.g. Windows)

    yield
    await batcher.stop()


app = FastAPI(title="BatchServe", lifespan=lifespan)


class PredictRequest(BaseModel):
    vector: list[float] | None = None  # optional; random vector used if omitted


class PredictResponse(BaseModel):
    score: float
    latency_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    payload = (
        np.array(req.vector, dtype=np.float32)
        if req.vector
        else np.random.default_rng().standard_normal(EMBED_DIM).astype(np.float32)
    )
    start = time.perf_counter()
    score = await batcher.submit(payload)
    elapsed = time.perf_counter() - start

    REQUEST_COUNT.inc()
    LATENCY.observe(elapsed)

    return PredictResponse(score=score, latency_ms=elapsed * 1000)


@app.get("/stats")
async def stats():
    sizes = batcher.stats["batch_sizes"]
    return {
        "total_requests": batcher.stats["requests"],
        "total_batches": batcher.stats["batches"],
        "avg_batch_size": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        "max_batch_size_seen": max(sizes) if sizes else 0,
        # last 60 real batch sizes, in order — used by the /  dashboard's
        # sparkline. Not a fabricated series: it's a direct slice of the
        # same list the benchmark and tests read from.
        "recent_batch_sizes": sizes[-60:],
    }


_DASHBOARD_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML


@app.get("/metrics")
async def metrics():
    for size in batcher.stats["batch_sizes"][-1:]:
        BATCH_SIZE_HIST.observe(size)
    BATCH_COUNT._value.set(batcher.stats["batches"])
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok"}
