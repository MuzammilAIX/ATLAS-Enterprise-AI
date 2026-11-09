"""
Initialize ATLAS tables in Neon PostgreSQL using infra/schema.sql.

Usage (from project root):
    python scripts/init_db.py
    python -m scripts.init_db          # also works

Safe to rerun:
    Existing tables and rows are preserved by CREATE TABLE IF NOT EXISTS.
"""

import sys
from pathlib import Path

# ── Path bootstrap: make project root importable ────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────

import psycopg
from psycopg.rows import dict_row

from app.config import settings


SCHEMA_FILE = PROJECT_ROOT / "infra" / "schema.sql"

REQUIRED_TABLES = (
    "documents",
    "document_versions",
    "feedback",
    "audit_events",
    "approvals",
    "chat_messages",
    "observability_events",
)


def verify_tables(conn) -> None:
    """Verify that all required ATLAS tables exist in public schema."""

    missing_tables = []
    for table in REQUIRED_TABLES:
        result = conn.execute(
            "SELECT to_regclass(%s) AS oid",
            (f"public.{table}",),
        ).fetchone()
        if result is None or result["oid"] is None:
            missing_tables.append(table)

    if missing_tables:
        raise RuntimeError(
            f"Missing required tables: {', '.join(missing_tables)}"
        )


def main() -> int:
    database_url = settings().database_url.strip()

    if not database_url or "YOUR_" in database_url.upper():
        print("ERROR: Set a valid DATABASE_URL in your project .env file.")
        return 1

    if not SCHEMA_FILE.is_file():
        print(f"ERROR: Schema file not found: {SCHEMA_FILE}")
        return 1

    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8").strip()
    if not schema_sql:
        print("ERROR: infra/schema.sql is empty.")
        return 1

    print("Connecting to ATLAS Neon PostgreSQL...")
    print("Database credentials are hidden.")

    try:
        with psycopg.connect(
            database_url,
            connect_timeout=15,
            application_name="atlas-db-initializer",
            row_factory=dict_row,          # ← FIX: enables db_info['db']
        ) as conn:

            with conn.transaction():
                conn.execute(schema_sql, prepare=False)

            db_info = conn.execute(
                """
                SELECT
                    current_database() AS db,
                    current_user      AS usr,
                    current_schema()  AS sch,
                    version()         AS ver
                """
            ).fetchone()

            verify_tables(conn)

        print("\nSUCCESS: ATLAS database initialized.")
        print(f"Database: {db_info['db']}")
        print(f"User:     {db_info['usr']}")
        print(f"Schema:   {db_info['sch']}")
        print(f"Verified tables: {len(REQUIRED_TABLES)}")
        for table in REQUIRED_TABLES:
            print(f"  [OK] {table}")

        print("\nExisting data has been preserved.")
        print("You can safely rerun this script.")
        return 0

    except Exception as exc:
        print("\nERROR: ATLAS database initialization failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Details: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())