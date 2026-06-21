#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Feast initialization entrypoint for Docker Compose / Kubernetes.
#
# 1. Wait for Redis + Postgres to be ready
# 2. Create the `feast` schema and offline store table in Postgres
# 3. Run `feast apply` to register feature definitions
# 4. Launch the Feast online feature server
# ---------------------------------------------------------------------------
set -euo pipefail

FEAST_REPO_PATH="${FEAST_REPO_PATH:-/feast_repo}"
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-face_recognition}"
POSTGRES_USER="${POSTGRES_USER:-admin}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-changeme123}"

echo "==> Installing feast dependencies..."
pip install --quiet "feast[redis,postgres]" redis structlog psycopg2-binary

# ── Wait for Redis ──────────────────────────────────────────────────────────
echo "==> Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
until python3 -c "import redis; redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}).ping()" 2>/dev/null; do
  echo "   Redis not ready – retrying in 3s..."
  sleep 3
done
echo "   Redis is ready."

# ── Wait for Postgres ───────────────────────────────────────────────────────
echo "==> Waiting for Postgres at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
until python3 -c "
import psycopg2
psycopg2.connect(host='${POSTGRES_HOST}', port=${POSTGRES_PORT}, dbname='${POSTGRES_DB}', user='${POSTGRES_USER}', password='${POSTGRES_PASSWORD}').close()
" 2>/dev/null; do
  echo "   Postgres not ready – retrying in 3s..."
  sleep 3
done
echo "   Postgres is ready."

# ── Create Feast schema + offline store table ───────────────────────────────
echo "==> Creating Feast schema and offline store table in Postgres..."
python3 - <<'PYEOF'
import psycopg2, os

conn = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "postgres"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    dbname=os.environ.get("POSTGRES_DB", "face_recognition"),
    user=os.environ.get("POSTGRES_USER", "admin"),
    password=os.environ.get("POSTGRES_PASSWORD", "changeme123"),
)
conn.autocommit = True
cur = conn.cursor()

# Create a dedicated schema for Feast offline store tables
cur.execute("CREATE SCHEMA IF NOT EXISTS feast;")
conn.close()
print("Feast schema and offline store table created successfully.")
PYEOF

# ── Run feast apply ─────────────────────────────────────────────────────────
echo "==> Running \`feast apply\` to register feature definitions..."
cd "${FEAST_REPO_PATH}"
feast apply

# ── Start Feast Feature Server ──────────────────────────────────────────────
echo "==> Starting Feast online feature server on port 6566..."
exec feast serve --host 0.0.0.0 --port 6566
