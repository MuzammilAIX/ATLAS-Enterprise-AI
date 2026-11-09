"""
S3 operations for document storage.

Credentials come from .env (local dev) or the AWS default chain
(EC2/ECS/Lambda). One key set without the other is a configuration error.
"""

from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.observability import trace


@lru_cache(maxsize=1)
def client():
    """Return a cached boto3 S3 client."""

    s = settings()

    has_key = bool(s.aws_access_key_id)
    has_secret = bool(s.aws_secret_access_key)

    if has_key != has_secret:
        raise ValueError(
            "Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or neither."
        )

    options: dict = {"region_name": s.aws_region}
    if has_key and has_secret:
        options["aws_access_key_id"] = s.aws_access_key_id
        options["aws_secret_access_key"] = s.aws_secret_access_key

    return boto3.client("s3", **options)


@trace("s3_upload")
def upload_bytes(
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to S3. Returns the s3_key."""

    bucket = settings().s3_bucket
    client().put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return key


@trace("s3_download")
def download_bytes(key: str) -> bytes:
    """Download raw bytes from S3."""

    bucket = settings().s3_bucket
    resp = client().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


@trace("s3_delete")
def delete_object(key: str) -> None:
    """Delete a single S3 object. No-op if it doesn't exist."""

    bucket = settings().s3_bucket
    client().delete_object(Bucket=bucket, Key=key)


def exists(key: str) -> bool:
    """Return True if the object exists in the configured bucket."""

    bucket = settings().s3_bucket
    try:
        client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL valid for `expires_in` seconds."""

    bucket = settings().s3_bucket
    return client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )