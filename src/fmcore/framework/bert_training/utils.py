"""
Utility functions for BERT training framework.

Provides helper functions for:
- Generating ISO timestamp run IDs
- Parsing S3 paths
- Other common utilities
"""

from datetime import datetime
from typing import Tuple


def generate_run_id() -> str:
    """
    Generate unique run ID based on UTC timestamp.

    Format: YYYY-MM-DDTHH-MM-SSZ (filesystem-safe, uses hyphens instead of colons)

    Returns:
        ISO timestamp string (e.g., "2024-11-24T14-30-45Z")

    Examples:
        >>> run_id = generate_run_id()
        >>> # Returns something like "2024-11-24T14-30-45Z"
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse S3 path into bucket and key components.

    Args:
        s3_path: Full S3 path (e.g., "s3://bucket-name/path/to/file.txt")

    Returns:
        Tuple of (bucket, key)
        - bucket: S3 bucket name
        - key: S3 object key (path within bucket)

    Raises:
        AssertionError: If path doesn't start with "s3://"

    Examples:
        >>> bucket, key = parse_s3_path("s3://my-bucket/data/file.parquet")
        >>> print(bucket)  # "my-bucket"
        >>> print(key)     # "data/file.parquet"
    """
    assert s3_path.startswith("s3://"), f"Invalid S3 path: {s3_path}"

    # Remove "s3://" prefix
    path_without_prefix = s3_path[5:]

    # Split on first "/" to separate bucket from key
    parts = path_without_prefix.split("/", 1)

    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    return bucket, key
