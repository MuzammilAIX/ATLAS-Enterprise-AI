"""
Text extraction from PDF / DOCX / TXT.

Called by worker/ingest.py. Given raw bytes + a mime/extension hint,
returns extracted plain text.

Extractors are best-effort: if pypdf can't extract (scanned PDF, image-only),
returns what it can rather than raising.
"""

from io import BytesIO
from pathlib import Path


def extract_text(
    data: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    Extract plain text from raw bytes.

    Detection order:
      1. content_type if provided
      2. filename extension
      3. fallback: try utf-8, then latin-1

    Raises ValueError if extraction fails hard (corrupt PDF).
    Returns empty string if nothing can be extracted (e.g. image-only PDF).
    """

    kind = _detect_kind(filename, content_type)

    if kind == "pdf":
        return _extract_pdf(data)
    if kind == "docx":
        return _extract_docx(data)
    if kind == "txt":
        return _extract_txt(data)

    return _extract_txt(data)


def _detect_kind(filename: str | None, content_type: str | None) -> str:
    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return "pdf"
        if "wordprocessingml" in ct or "msword" in ct:
            return "docx"
        if ct.startswith("text/"):
            return "txt"

    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext in (".docx", ".doc"):
            return "docx"
        if ext in (".txt", ".md", ".csv"):
            return "txt"

    return "unknown"


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts).strip()


def _extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace").strip()