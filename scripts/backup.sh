#!/bin/bash
# Database backup script for OpenManus

set -e

# Configuration
BACKUP_DIR="/backup"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="openmanus_backup_${TIMESTAMP}.sql"
RETENTION_DAYS=7

# Database connection settings (from environment)
PGHOST=${POSTGRES_HOST:-postgres}
PGPORT=${POSTGRES_PORT:-5432}
PGDATABASE=${POSTGRES_DB:-openmanus}
PGUSER=${POSTGRES_USER:-openmanus}

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

echo "Starting database backup at $(date)"
echo "Database: ${PGDATABASE} on ${PGHOST}:${PGPORT}"
echo "Backup file: ${BACKUP_FILE}"

# Create database dump
pg_dump \
    --host="${PGHOST}" \
    --port="${PGPORT}" \
    --username="${PGUSER}" \
    --dbname="${PGDATABASE}" \
    --verbose \
    --clean \
    --if-exists \
    --create \
    --format=plain \
    --file="${BACKUP_DIR}/${BACKUP_FILE}"

# Compress the backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

echo "Backup completed: ${COMPRESSED_FILE}"
echo "Backup size: $(du -h "${BACKUP_DIR}/${COMPRESSED_FILE}" | cut -f1)"

# Clean up old backups (keep only last N days)
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "openmanus_backup_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete

# List remaining backups
echo "Available backups:"
ls -lh "${BACKUP_DIR}"/openmanus_backup_*.sql.gz 2>/dev/null || echo "No backups found"

echo "Backup process completed at $(date)"
