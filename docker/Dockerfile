# syntax=docker/dockerfile:1
# ── Phase 1 smoke-test Dockerfile ─────────────────────────────────────────────
# Purpose: prove the container boots and Streamlit is reachable.
# This is NOT a production image. Hardening (non-root user, slim base,
# model pre-baking) is deferred to Phase 5.
#
# Build:  docker build -f docker/Dockerfile -t sih-alpha .
# Run:    docker run -p 8501:8501 sih-alpha
# Test:   open http://localhost:8501

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer-cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: disable telemetry, bind to all interfaces
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Generate dummy forecast if the real one is not present
RUN python dummy_generator.py

CMD ["streamlit", "run", "src/ui/app.py", "--server.headless=true"]
