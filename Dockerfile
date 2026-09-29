FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first
COPY pyproject.toml uv.lock README.md ./

# Install runtime dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ ./src/

# Install our Python package
RUN uv sync --frozen --no-dev

# Copy trained inference artifacts
COPY artifacts/mf/ ./artifacts/mf/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", \
     "recommender_system.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]