"""
Split extracted text into overlapping chunks for embedding.

Chunk size and overlap are tuned for BAAI/bge-small-en-v1.5 (384-dim):
~800 chars per chunk captures enough context without exceeding the model's
sweet spot. 100-char overlap preserves meaning across chunk boundaries.
"""


DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Prefers to break at paragraph / sentence boundaries when possible
    to avoid cutting mid-sentence. Falls back to a hard cut otherwise.

    Returns an empty list for empty/whitespace-only text.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    # Short text → single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # If not at the end, try to break at a paragraph or sentence boundary
        if end < n:
            window = text[start:end]
            # Prefer double-newline (paragraph)
            para_break = window.rfind("\n\n")
            if para_break > chunk_size // 2:
                end = start + para_break + 2
            else:
                # Prefer period / question / exclamation + space
                for punct in (". ", "? ", "! "):
                    sent_break = window.rfind(punct)
                    if sent_break > chunk_size // 2:
                        end = start + sent_break + len(punct)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # Next chunk starts `overlap` chars before current end (to preserve context)
        start = max(end - overlap, start + 1)

    return chunks