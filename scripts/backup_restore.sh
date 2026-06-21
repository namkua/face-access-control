#!/usr/bin/env bash
# =============================================================================
# backup_restore.sh – Automated backup & restore for PostgreSQL and MinIO
# =============================================================================
# Usage:
#   ./scripts/backup_restore.sh backup          # backup all
#   ./scripts/backup_restore.sh backup-db       # backup PostgreSQL only
#   ./scripts/backup_restore.sh backup-minio    # backup MinIO only
#   ./scripts/backup_restore.sh restore-db  <file.sql.gz>
#   ./scripts/backup_restore.sh restore-minio <bucket> <archive.tar.gz>
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-face_recognition}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-secret}"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_BUCKETS="${MINIO_BUCKETS:-raw-images processed-data}"

BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" &>/dev/null || die "'$1' is not installed."
}

ensure_backup_dir() {
  mkdir -p "${BACKUP_DIR}/db" "${BACKUP_DIR}/minio"
  log "Backup directory: ${BACKUP_DIR}"
}

# ---------------------------------------------------------------------------
# PostgreSQL backup
# ---------------------------------------------------------------------------
backup_db() {
  require_cmd pg_dump
  require_cmd gzip

  local out="${BACKUP_DIR}/db/${POSTGRES_DB}_${TIMESTAMP}.sql.gz"
  log "Backing up PostgreSQL → ${out}"

  PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    -h "${POSTGRES_HOST}" \
    -p "${POSTGRES_PORT}" \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    --no-password \
    --format=plain \
    --verbose \
    | gzip > "${out}"

  log "PostgreSQL backup complete: ${out} ($(du -sh "${out}" | cut -f1))"
}

# ---------------------------------------------------------------------------
# PostgreSQL restore
# ---------------------------------------------------------------------------
restore_db() {
  local dump_file="${1:-}"
  [[ -z "${dump_file}" ]] && die "Usage: $0 restore-db <file.sql.gz>"
  [[ -f "${dump_file}" ]] || die "File not found: ${dump_file}"

  require_cmd psql
  require_cmd gunzip

  log "Restoring PostgreSQL from ${dump_file}"

  PGPASSWORD="${POSTGRES_PASSWORD}" gunzip -c "${dump_file}" | psql \
    -h "${POSTGRES_HOST}" \
    -p "${POSTGRES_PORT}" \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    --no-password

  log "PostgreSQL restore complete."
}

# ---------------------------------------------------------------------------
# MinIO backup (using mc mirror)
# ---------------------------------------------------------------------------
backup_minio() {
  require_cmd mc

  # Configure mc alias
  mc alias set backup_src "${MINIO_ENDPOINT}" \
    "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" --insecure 2>/dev/null || true

  for bucket in ${MINIO_BUCKETS}; do
    local out="${BACKUP_DIR}/minio/${bucket}_${TIMESTAMP}.tar.gz"
    local tmp_dir="${BACKUP_DIR}/minio/.tmp_${bucket}_${TIMESTAMP}"

    log "Backing up MinIO bucket '${bucket}' → ${out}"
    mkdir -p "${tmp_dir}"

    mc mirror --overwrite "backup_src/${bucket}" "${tmp_dir}"
    tar -czf "${out}" -C "${tmp_dir}" .
    rm -rf "${tmp_dir}"

    log "MinIO backup complete: ${out} ($(du -sh "${out}" | cut -f1))"
  done
}

# ---------------------------------------------------------------------------
# MinIO restore
# ---------------------------------------------------------------------------
restore_minio() {
  local bucket="${1:-}"
  local archive="${2:-}"
  [[ -z "${bucket}" || -z "${archive}" ]] && die "Usage: $0 restore-minio <bucket> <archive.tar.gz>"
  [[ -f "${archive}" ]] || die "File not found: ${archive}"

  require_cmd mc
  require_cmd tar

  mc alias set restore_dst "${MINIO_ENDPOINT}" \
    "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" --insecure 2>/dev/null || true

  local tmp_dir="${BACKUP_DIR}/minio/.restore_${TIMESTAMP}"
  mkdir -p "${tmp_dir}"

  log "Extracting ${archive} → ${tmp_dir}"
  tar -xzf "${archive}" -C "${tmp_dir}"

  log "Uploading to MinIO bucket '${bucket}'"
  mc mirror --overwrite "${tmp_dir}/" "restore_dst/${bucket}"

  rm -rf "${tmp_dir}"
  log "MinIO restore complete."
}

# ---------------------------------------------------------------------------
# Cleanup old backups
# ---------------------------------------------------------------------------
cleanup_old_backups() {
  log "Removing backups older than ${RETENTION_DAYS} days..."
  find "${BACKUP_DIR}" -type f \( -name "*.gz" -o -name "*.tar.gz" \) \
    -mtime "+${RETENTION_DAYS}" -delete
  log "Cleanup done."
}

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
CMD="${1:-help}"

case "${CMD}" in
  backup)
    ensure_backup_dir
    backup_db
    backup_minio
    cleanup_old_backups
    ;;
  backup-db)
    ensure_backup_dir
    backup_db
    ;;
  backup-minio)
    ensure_backup_dir
    backup_minio
    ;;
  restore-db)
    restore_db "${2:-}"
    ;;
  restore-minio)
    restore_minio "${2:-}" "${3:-}"
    ;;
  cleanup)
    cleanup_old_backups
    ;;
  *)
    echo "Usage: $0 {backup|backup-db|backup-minio|restore-db <file>|restore-minio <bucket> <file>|cleanup}"
    exit 1
    ;;
esac
