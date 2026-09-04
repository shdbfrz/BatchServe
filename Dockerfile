FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# uvicorn forwards SIGTERM to the app's lifespan shutdown, which drains
# in-flight batches (see app/main.py -> lifespan -> batcher.stop()).
# --timeout-graceful-shutdown gives it headroom to finish before the
# container runtime escalates to SIGKILL.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "10"]
