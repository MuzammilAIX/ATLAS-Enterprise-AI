"""Quick script to inspect observability_events. Temporary — delete when done."""

import sys
from pathlib import Path

# ── Path bootstrap: make project root importable ────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────

from app.db import connection


with connection() as conn:
    rows = conn.execute(
        """
        SELECT
            event_type,
            tenant_id,
            actor,
            detail->>'status'      AS status,
            detail->>'duration_ms' AS ms
        FROM observability_events
        WHERE event_type LIKE 'test_%'
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()

for r in rows:
    print(r)