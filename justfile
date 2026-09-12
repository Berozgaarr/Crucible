# Crucible Justfile — Developer Command Runner

set dotenv-load := false
set shell := ["bash", "-uc"]

port := "8000"
ui_port := "5173"
host := "0.0.0.0"

# Show available commands
default:
    @just --list

# Install all backend and frontend dependencies
install:
    uv sync
    bun --cwd frontend install

# Run full development stack concurrently
dev:
    @echo "Starting Studio (API :{{port}} + UI :{{ui_port}})..."
    @trap 'kill 0' EXIT; \
    uv run uvicorn crucible.app:app --host {{host}} --port {{port}} --reload & \
    bun --cwd frontend dev --port {{ui_port}} --host {{host}}

# Run full development stack and expose via bore public tunnel
share:
    @echo "Starting Studio and bore tunnel..."
    @trap 'kill 0' EXIT; \
    uv run uvicorn crucible.app:app --host {{host}} --port {{port}} --reload & \
    bun --cwd frontend dev --port {{ui_port}} --host {{host}} & \
    bore local {{ui_port}} --to bore.pub 2>&1 | sed -nu 's/.*listening at \([^ ]*\).*/\n🔗 PUBLIC SHARE URL: http:\/\/\1\n/p'



# Start backend JSON API server only
api:
    uv run uvicorn crucible.app:app --host {{host}} --port {{port}} --reload

# Start SvelteKit frontend dev server only
ui:
    bun --cwd frontend dev --port {{ui_port}} --host {{host}}

# Build SvelteKit frontend for production
build:
    bun --cwd frontend run build

# Run tests (e.g. just test, just test -k parser, just test -x)
test *args="":
    uv run pytest tests/ {{args}}

# Run end-to-end pipeline smoke test
smoke:
    uv run python -m crucible.scripts.smoke_test

# Format Python code
format:
    uv run ruff format crucible/ tests/
    uv run ruff check --fix crucible/ tests/

# Lint & typecheck backend and frontend
check:
    uv run ruff check crucible/ tests/
    uv run ruff format --check crucible/ tests/
    bun --cwd frontend run check

# Regenerate offline demo candidates pre-cache
cache:
    HF_HUB_OFFLINE=1 uv run python -m crucible.scripts.generate_cache


# Clean build artifacts, caches, and bytecode
clean:
    rm -rf .pytest_cache .ruff_cache htmlcov .coverage frontend/.svelte-kit frontend/build frontend/node_modules/.vite
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.py[co]" -delete


