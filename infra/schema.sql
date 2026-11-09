-- ==========================================
-- ATLAS ENTERPRISE AI
-- PostgreSQL / Neon Database Schema
-- ==========================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ==========================================
-- 1. DOCUMENTS
-- ==========================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    owner_sub TEXT NOT NULL,

    allowed_groups TEXT[] NOT NULL DEFAULT '{}',

    s3_key TEXT NOT NULL UNIQUE,

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'processing',
                'ready',
                'failed',
                'retired'
            )
        ),

    current_version INTEGER NOT NULL DEFAULT 1
        CHECK (current_version >= 1),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_acl_idx
ON documents (tenant_id, status);

CREATE INDEX IF NOT EXISTS documents_owner_idx
ON documents (tenant_id, owner_sub);

CREATE INDEX IF NOT EXISTS documents_groups_idx
ON documents USING GIN (allowed_groups);


-- ==========================================
-- 2. DOCUMENT VERSIONS
-- ==========================================

CREATE TABLE IF NOT EXISTS document_versions (
    id BIGSERIAL PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,

    version INTEGER NOT NULL CHECK (version >= 1),

    s3_key TEXT NOT NULL UNIQUE,
    content_hash TEXT,

    created_by TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (document_id, version)
);

CREATE INDEX IF NOT EXISTS document_versions_doc_idx
ON document_versions (document_id, version DESC);


-- ==========================================
-- 3. USER FEEDBACK
-- ==========================================

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,

    tenant_id TEXT NOT NULL,
    actor TEXT NOT NULL,

    conversation_id UUID,

    rating INTEGER NOT NULL
        CHECK (rating BETWEEN -1 AND 1),

    note TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS feedback_tenant_time_idx
ON feedback (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS feedback_conversation_idx
ON feedback (tenant_id, conversation_id)
WHERE conversation_id IS NOT NULL;


-- ==========================================
-- 4. AUDIT EVENTS
-- ==========================================

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,

    actor TEXT NOT NULL,
    tenant_id TEXT NOT NULL,

    action TEXT NOT NULL,

    detail JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS audit_tenant_time_idx
ON audit_events (tenant_id, created_at DESC);


-- ==========================================
-- 5. HUMAN-IN-THE-LOOP APPROVALS
-- ==========================================

CREATE TABLE IF NOT EXISTS approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id TEXT NOT NULL,

    requester TEXT NOT NULL,
    question TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'approved',
                'rejected',
                'cancelled'
            )
        ),

    reviewer TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,

    CHECK (
        (status = 'pending' AND reviewed_at IS NULL)
        OR
        (status <> 'pending' AND reviewed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS approvals_tenant_idx
ON approvals (tenant_id, status);


-- ==========================================
-- 6. CONVERSATION MEMORY
-- ==========================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,

    tenant_id TEXT NOT NULL,

    conversation_id UUID NOT NULL,

    role TEXT NOT NULL
        CHECK (role IN ('user', 'assistant')),

    content TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx
ON chat_messages (
    tenant_id,
    conversation_id,
    id DESC
);


-- ==========================================
-- 7. OBSERVABILITY
-- ==========================================

CREATE TABLE IF NOT EXISTS observability_events (
    id BIGSERIAL PRIMARY KEY,

    tenant_id TEXT NOT NULL,
    actor TEXT,

    trace_id TEXT,
    event_type TEXT NOT NULL,

    detail JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS observability_tenant_time_idx
ON observability_events (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS observability_trace_idx
ON observability_events (trace_id)
WHERE trace_id IS NOT NULL;


-- ==========================================
-- SCHEMA INITIALIZATION COMPLETE
-- ==========================================