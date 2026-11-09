"""Tests for app.memory — conversation persistence and retrieval."""

import uuid

import pytest

from app.db import connection
from app.memory import (
    clear_conversation,
    get_recent,
    save_assistant_message,
    save_user_message,
)


@pytest.fixture
def tenant_and_conv():
    """Unique tenant + conversation per test — auto-cleaned at teardown."""
    tenant = f"test_tenant_{uuid.uuid4().hex[:8]}"
    conv = str(uuid.uuid4())
    yield tenant, conv
    with connection() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE tenant_id = %s AND conversation_id = %s",
            (tenant, conv),
        )


def test_save_and_get_user_message(tenant_and_conv):
    tenant, conv = tenant_and_conv
    save_user_message(tenant, conv, "hello")
    msgs = get_recent(tenant, conv)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"


def test_save_both_roles_preserves_order(tenant_and_conv):
    tenant, conv = tenant_and_conv
    save_user_message(tenant, conv, "q1")
    save_assistant_message(tenant, conv, "a1")
    save_user_message(tenant, conv, "q2")
    save_assistant_message(tenant, conv, "a2")

    msgs = get_recent(tenant, conv)
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_get_recent_limits_to_n_turns(tenant_and_conv):
    tenant, conv = tenant_and_conv
    for i in range(5):
        save_user_message(tenant, conv, f"u{i}")
        save_assistant_message(tenant, conv, f"a{i}")

    msgs = get_recent(tenant, conv, limit_turns=2)
    assert len(msgs) == 4
    assert [m["content"] for m in msgs] == ["u3", "a3", "u4", "a4"]


def test_get_recent_defaults_to_config_turns(tenant_and_conv):
    tenant, conv = tenant_and_conv
    from app.config import settings
    n = settings().memory_turns

    for i in range(n + 2):
        save_user_message(tenant, conv, f"u{i}")
        save_assistant_message(tenant, conv, f"a{i}")

    msgs = get_recent(tenant, conv)
    assert len(msgs) == n * 2


def test_get_recent_empty_conversation(tenant_and_conv):
    tenant, conv = tenant_and_conv
    msgs = get_recent(tenant, conv)
    assert msgs == []


def test_get_recent_is_tenant_scoped(tenant_and_conv):
    tenant, conv = tenant_and_conv
    other_tenant = f"other_{uuid.uuid4().hex[:8]}"
    other_conv = str(uuid.uuid4())

    save_user_message(tenant, conv, "mine")
    save_user_message(other_tenant, other_conv, "not mine")

    try:
        msgs = get_recent(tenant, conv)
        assert [m["content"] for m in msgs] == ["mine"]
    finally:
        with connection() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE tenant_id = %s AND conversation_id = %s",
                (other_tenant, other_conv),
            )


def test_clear_conversation_removes_messages(tenant_and_conv):
    tenant, conv = tenant_and_conv
    save_user_message(tenant, conv, "hi")
    save_assistant_message(tenant, conv, "hello")

    deleted = clear_conversation(tenant, conv)
    assert deleted == 2

    assert get_recent(tenant, conv) == []


def test_save_rejects_empty_content(tenant_and_conv):
    tenant, conv = tenant_and_conv

    with pytest.raises(ValueError, match="empty"):
        save_user_message(tenant, conv, "")

    with pytest.raises(ValueError, match="empty"):
        save_user_message(tenant, conv, "   ")


def test_invalid_limit_turns_raises(tenant_and_conv):
    tenant, conv = tenant_and_conv
    with pytest.raises(ValueError, match="limit_turns"):
        get_recent(tenant, conv, limit_turns=0)
    with pytest.raises(ValueError, match="limit_turns"):
        get_recent(tenant, conv, limit_turns=-1)