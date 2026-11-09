"""
Guardrails — validate and sanitize untrusted strings before they reach
the LLM or Pinecone.

Every user question and every retrieved chunk passes through this module.
Cheap, deterministic, no network, no DB. This is the security boundary.
"""

import re
import unicodedata


MAX_QUESTION_CHARS = 2000
MAX_TENANT_CHARS = 64
MAX_CHUNK_CHARS = 8000  # for evidence injection

_TENANT_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Heuristics for prompt injection — matches common attack phrases.
# Not exhaustive, but catches the obvious attempts that a malicious
# document or user would embed.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        re.I,
    ),
    re.compile(
        r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        re.I,
    ),
    re.compile(
        r"\bforget\s+(everything|all|your)\s+(above|previous|prior|instructions?|rules?)",
        re.I,
    ),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bact\s+as\s+(a\s+|an\s+)?(?!the\s+assistant)", re.I),
    # "System:" at the start of a line (multiline) — classic role-override.
    # Also catches "System: prompt", "System: override", "System: message".
    re.compile(r"^\s*system\s*:", re.I | re.M),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(
        r"<\s*\|\s*(im_start|im_end|system|user|assistant)\s*\|\s*>",
        re.I,
    ),
    re.compile(r"\[\s*(SYSTEM|INST|/INST)\s*\]", re.I),
    re.compile(r"\breveal\s+your\s+(system\s+)?prompt", re.I),
    re.compile(r"\bprint\s+(your\s+)?(system\s+)?prompt", re.I),
]


# ==========================================
# QUESTION VALIDATION
# ==========================================

def validate_question(text: str) -> str:
    """
    Validate and normalize a user question.

    Raises ValueError if:
      - Empty / whitespace-only
      - Longer than MAX_QUESTION_CHARS
      - Contains NUL or control chars (excluding \\n \\r \\t)

    Returns the trimmed, control-char-stripped text.
    """

    if not isinstance(text, str):
        raise ValueError("question must be a string")

    if not text.strip():
        raise ValueError("question cannot be empty")

    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question too long ({len(text)} > {MAX_QUESTION_CHARS} chars)"
        )

    if "\x00" in text:
        raise ValueError("question contains NUL byte")

    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t":
            raise ValueError(f"question contains control char: {ch!r}")

    return text.strip()


# ==========================================
# PROMPT INJECTION DETECTION
# ==========================================

def detect_prompt_injection(text: str) -> list[str]:
    """
    Return a list of matched injection pattern descriptions.
    Empty list = clean.

    Used for logging only — do NOT block on this alone. Combine with
    model-side defense (system prompt hardening) and content filtering.
    """

    if not isinstance(text, str):
        return []

    hits = []
    for i, pat in enumerate(_INJECTION_PATTERNS):
        m = pat.search(text)
        if m:
            hits.append(f"pattern_{i}: {m.group(0)!r}")

    return hits


def is_suspicious(text: str) -> bool:
    """Convenience wrapper: True if any injection pattern matches."""
    return bool(detect_prompt_injection(text))


# ==========================================
# IDENTIFIER VALIDATION
# ==========================================

def validate_tenant_id(tenant_id: str) -> str:
    """
    Tenant IDs: alphanumeric + `_` + `-`, 1–64 chars.

    Prevents path traversal, SQL-adjacent injection via IDs,
    and mistakes like passing a full UUID when a short slug was expected.
    """

    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id must be a string")

    if not _TENANT_RE.match(tenant_id):
        raise ValueError(
            f"invalid tenant_id: {tenant_id!r} "
            "(must be 1-64 chars, alphanumeric + _ + -)"
        )

    return tenant_id


def validate_conversation_id(conv_id: str) -> str:
    """Conversation IDs must be UUID strings."""

    if not isinstance(conv_id, str):
        raise ValueError("conversation_id must be a string")

    if not _UUID_RE.match(conv_id):
        raise ValueError(f"invalid conversation_id: {conv_id!r} (must be a UUID)")

    return conv_id


# ==========================================
# EVIDENCE / CHUNK SANITIZATION
# ==========================================

def sanitize_for_evidence(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """
    Sanitize a retrieved chunk before injecting into the LLM prompt.

    - Strips NUL and non-newline control chars
    - Truncates to max_chars
    - Does NOT strip injection patterns (the LLM system prompt must
      handle those, because stripping them silently could break
      legitimate content that merely mentions "ignore previous").

    Returns sanitized text.
    """

    if not isinstance(text, str):
        raise ValueError("chunk text must be a string")

    cleaned = []
    for ch in text:
        if ch == "\x00":
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in "\n\r\t":
            continue
        cleaned.append(ch)

    result = "".join(cleaned)

    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "…"

    return result


# ==========================================
# COMBINED ENTRY POINT
# ==========================================

def check_question(text: str) -> dict:
    """
    Full validation pipeline for a user question.

    Returns:
        {
            "clean": str,            # sanitized question
            "suspicious": bool,      # prompt-injection pattern matched?
            "hits": list[str],       # pattern descriptions
        }

    Raises ValueError on structural problems (empty, too long, control chars).
    """

    clean = validate_question(text)
    hits = detect_prompt_injection(clean)
    return {
        "clean": clean,
        "suspicious": bool(hits),
        "hits": hits,
    }