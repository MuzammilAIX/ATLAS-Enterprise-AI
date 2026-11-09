"""
ATLAS — Integration Smoke Test

Verifies that all 4 external services are reachable and configured
correctly BEFORE any pipeline code runs:

    [1] OpenAI       — chat completion call
    [2] Pinecone     — connect, list indexes, check target index
    [3] AWS S3       — bucket exists in correct region
    [4] Neon Postgres — DB reachable, 7 tables present

Usage (from project root):
    python -m scripts.smoke_test
    python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

# ── Path bootstrap: make project root importable ────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────

from app.config import settings
from app.db import check_database, connection


PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "


# ==========================================
# HELPERS
# ==========================================

def _banner(index: int, total: int, name: str) -> None:
    print(f"\n[{index}/{total}] {name} ", end="", flush=True)
    dots = "." * max(2, 30 - len(name))
    print(dots, end=" ", flush=True)


def _ok(msg: str) -> bool:
    print(f"{PASS}  {msg}")
    return True


def _fail(msg: str) -> bool:
    print(f"{FAIL}  {msg}")
    return False


# ==========================================
# [1] OPENAI
# ==========================================

def check_openai(cfg) -> bool:
    try:
        from openai import OpenAI
    except ImportError:
        return _fail("openai package not installed — pip install openai")

    if not cfg.openai_api_key.startswith("sk-"):
        return _fail(
            f"OPENAI_API_KEY missing or placeholder (starts with "
            f"{cfg.openai_api_key[:6]!r})"
        )

    try:
        client = OpenAI(api_key=cfg.openai_api_key)
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[{"role": "user", "content": "Reply with just: ok"}],
            max_tokens=10,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        return _ok(f"{cfg.openai_model}, {elapsed_ms} ms, reply={text!r}")
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


# ==========================================
# [2] PINECONE
# ==========================================

def check_pinecone(cfg) -> bool:
    try:
        from pinecone import Pinecone
    except ImportError:
        return _fail("pinecone package not installed — pip install pinecone")

    if not cfg.pinecone_api_key:
        return _fail("PINECONE_API_KEY is empty")

    try:
        pc = Pinecone(api_key=cfg.pinecone_api_key)
        indexes = pc.list_indexes()
        names = list(indexes.names())  # v8 API: returns IndexList with .names()

        if cfg.pinecone_index_name not in names:
            return _fail(
                f"index {cfg.pinecone_index_name!r} NOT FOUND. "
                f"Existing indexes: {names or '(none)'}. "
                f"Create it with dimension={cfg.embed_dim}, metric='cosine'."
            )

        index = pc.Index(cfg.pinecone_index_name)
        stats = index.describe_index_stats()
        dim = stats.dimension
        count = stats.total_vector_count

        if dim != cfg.embed_dim:
            return _fail(
                f"index {cfg.pinecone_index_name!r} has dimension={dim}, "
                f"but EMBED_DIM={cfg.embed_dim}. They must match."
            )

        return _ok(
            f"{cfg.pinecone_index_name!r}, {dim} dim, {count} vectors"
        )
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


# ==========================================
# [3] AWS S3
# ==========================================

def check_s3(cfg) -> bool:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return _fail("boto3 not installed — pip install boto3")

    if not cfg.s3_bucket:
        return _fail("S3_BUCKET is empty")

    try:
        # If keys are set, pass explicitly; otherwise rely on default chain
        if cfg.aws_access_key_id and cfg.aws_secret_access_key:
            session = boto3.Session(
                aws_access_key_id=cfg.aws_access_key_id,
                aws_secret_access_key=cfg.aws_secret_access_key,
                region_name=cfg.aws_region,
            )
        else:
            session = boto3.Session(region_name=cfg.aws_region)

        s3 = session.client("s3")
        s3.head_bucket(Bucket=cfg.s3_bucket)

        # Confirm region matches
        loc = s3.get_bucket_location(Bucket=cfg.s3_bucket).get(
            "LocationConstraint"
        )
        # S3 returns None for us-east-1; our region should match
        if loc not in (None, cfg.aws_region):
            return _fail(
                f"bucket {cfg.s3_bucket!r} is in {loc!r}, "
                f"but AWS_REGION={cfg.aws_region!r}"
            )

        return _ok(f"{cfg.s3_bucket!r}, {cfg.aws_region}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            return _fail(f"bucket {cfg.s3_bucket!r} does not exist")
        if code in ("403", "AccessDenied"):
            return _fail(f"credentials rejected for bucket {cfg.s3_bucket!r}")
        return _fail(f"{code}: {exc}")
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


# ==========================================
# [4] NEON POSTGRES
# ==========================================

def check_postgres(cfg) -> bool:
    if not check_database():
        return _fail("check_database() returned False")

    try:
        expected = {
            "documents", "document_versions", "feedback",
            "audit_events", "approvals", "chat_messages",
            "observability_events",
        }
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            ).fetchall()
        actual = {r["table_name"] for r in rows}
        missing = expected - actual
        if missing:
            return _fail(f"missing tables: {sorted(missing)}")
        return _ok(f"{len(expected)} tables present")
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


# ==========================================
# MAIN
# ==========================================

def main() -> int:
    cfg = settings()
    results: list[tuple[str, bool]] = []

    print("=" * 60)
    print("ATLAS — Integration Smoke Test")
    print("=" * 60)

    _banner(1, 4, "OpenAI")
    results.append(("OpenAI", check_openai(cfg)))

    _banner(2, 4, "Pinecone")
    results.append(("Pinecone", check_pinecone(cfg)))

    _banner(3, 4, "AWS S3")
    results.append(("S3", check_s3(cfg)))

    _banner(4, 4, "Neon Postgres")
    results.append(("Postgres", check_postgres(cfg)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    failed = 0
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed. Fix them before running the pipeline.")
        return 1

    print("\nAll integrations healthy. Ready for the pipeline layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())