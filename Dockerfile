FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PORT=8000 \
    SOP_DB_PATH=/data/sop.db \
    SOP_OUTBOX_DIR=/data/outbox

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sop_agent ./sop_agent
COPY fixtures ./fixtures

# Run as an unprivileged user; conversations, lockout counters and the outbox live on /data.
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# Model key, DEV_PASSWORD etc. come from the environment at runtime (never baked into the image).
CMD ["sh", "-c", "uvicorn sop_agent.server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
