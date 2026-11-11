// ATLAS single-page UI — wired to /query, /upload, /documents, /api/approvals.

const $ = id => document.getElementById(id);

// ── UUID generator (works on HTTP; crypto.randomUUID is HTTPS-only) ──
function uuidv4() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        try {
            return window.crypto.randomUUID();
        } catch (e) {
            // fall through
        }
    }
    if (window.crypto && typeof window.crypto.getRandomValues === 'function') {
        const bytes = new Uint8Array(16);
        window.crypto.getRandomValues(bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40;  // version 4
        bytes[8] = (bytes[8] & 0x3f) | 0x80;  // variant RFC 4122
        const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
        return (
            hex.slice(0, 8) + '-' +
            hex.slice(8, 12) + '-' +
            hex.slice(12, 16) + '-' +
            hex.slice(16, 20) + '-' +
            hex.slice(20, 32)
        );
    }
    // Last resort — Math.random (weaker but works everywhere)
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

let conversationId = sessionStorage.getItem('atlas_conversation_id') || uuidv4();
sessionStorage.setItem('atlas_conversation_id', conversationId);

// ── Toast ─────────────────────────────────────────
const toast = msg => {
    const el = $('toast');
    if (!el) return;
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.style.display = 'none'; }, 4000);
};

// ── Fetch helper ──────────────────────────────────
const api = async (path, options = {}) => {
    const res = await fetch(path, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    if (!res.ok) throw Error(data.detail || ('HTTP ' + res.status));
    return data;
};

// ── Tab routing ───────────────────────────────────
function view(name) {
    ['chat', 'docs', 'approval', 'admin'].forEach(x => {
        const v = $(x + 'View');
        const t = $(x + 'Tab');
        if (v) v.classList.toggle('hidden', x !== name);
        if (t) t.classList.toggle('active', x === name);
    });
    const h = $('heading');
    if (h) {
        h.textContent = ({
            chat: "Ask your organization's knowledge",
            docs: 'Document management',
            approval: 'Approval requests',
            admin: 'Operational overview',
        })[name] || '';
    }
    if (name === 'docs') loadDocs();
    if (name === 'admin') loadMetrics();
    if (name === 'approval') loadApprovals();
}

// ── Wire up tabs safely (only if elements exist) ──
['chat', 'docs', 'admin', 'approval'].forEach(x => {
    const t = $(x + 'Tab');
    if (t) t.onclick = () => view(x);
});

const _newChat = $('newChat');
if (_newChat) {
    _newChat.onclick = () => {
        conversationId = uuidv4();
        sessionStorage.setItem('atlas_conversation_id', conversationId);
        const msgs = $('messages');
        if (msgs) msgs.replaceChildren();
        const hero = $('heroBlock');
        if (hero) hero.classList.remove('hidden');
        toast('Started a new conversation');
    };
}

const _refresh = $('refreshApprovals');
if (_refresh) _refresh.onclick = loadApprovals;

// ── Chat ──────────────────────────────────────────
function addMessage(content, kind, sources = [], extra = null) {
    const messages = $('messages');
    if (!messages) return;

    const el = document.createElement('div');
    el.className = 'message ' + kind;
    el.textContent = content;

    if (sources && sources.length) {
        const wrap = document.createElement('div');
        wrap.className = 'sources';
        sources.forEach((s, i) => {
            const d = document.createElement('details');
            const title = document.createElement('summary');
            const score = (s.score != null ? s.score : 0).toFixed(3);
            const docId = (s.metadata && s.metadata.document_id) || 'unknown';
            title.textContent = `[${i + 1}] doc=${docId} · score=${score}`;
            const excerpt = document.createElement('p');
            excerpt.textContent = ((s.metadata && s.metadata.text) || '').slice(0, 400);
            d.append(title, excerpt);
            wrap.append(d);
        });
        el.append(wrap);
    }

    if (extra && extra.approval_id) {
        const info = document.createElement('div');
        info.style.marginTop = '10px';
        info.style.fontSize = '12px';
        info.style.color = '#a8c3dc';
        info.textContent = `Approval ID: ${extra.approval_id.slice(0, 8)}... · See Approvals tab`;
        el.append(info);
    }

    messages.append(el);
    messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });
}

const chatForm = $('chatForm');
if (chatForm) {
    chatForm.onsubmit = async e => {
        e.preventDefault();
        const q = $('question').value.trim();
        if (!q) return;

        addMessage(q, 'user');
        $('question').value = '';
        const hero = $('heroBlock');
        if (hero) hero.classList.add('hidden');

        const messages = $('messages');
        const loading = document.createElement('div');
        loading.className = 'message assistant';
        loading.textContent = 'Searching approved documents...';
        messages.append(loading);
        messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });

        try {
            const r = await api('/query', {
                method: 'POST',
                body: JSON.stringify({
                    question: q,
                    tenant_id: ($('tenantInput').value.trim() || 'demo'),
                    conversation_id: conversationId,
                    safe_mode: $('safeMode').checked,
                    actor: 'demo-user',
                }),
            });
            loading.remove();

            if (r.needs_approval) {
                addMessage(
                    'This question requires human approval before I can answer it.\n\n' +
                    'Reason: ' + (r.approval_reason || 'sensitive content'),
                    'needs-approval',
                    [],
                    { approval_id: r.approval_id }
                );
            } else {
                addMessage(r.answer || '(no answer)', 'assistant', r.sources || []);
            }
        } catch (err) {
            loading.remove();
            addMessage('Error: ' + err.message, 'error');
        }
    };
}

// ── Suggestion buttons ────────────────────────────
document.querySelectorAll('[data-q]').forEach(b => {
    b.onclick = () => {
        const q = $('question');
        if (q) { q.value = b.dataset.q; q.focus(); }
    };
});

// ── Documents ─────────────────────────────────────
async function loadDocs() {
    const root = $('docs');
    if (!root) return;
    root.textContent = 'Loading...';
    try {
        const tenant = ($('tenantInput').value.trim() || 'demo');
        const r = await api(`/documents/${encodeURIComponent(tenant)}`);
        root.replaceChildren();

        if (!r.documents || r.documents.length === 0) {
            root.textContent = 'No documents yet. Upload one above.';
            return;
        }

        for (const d of r.documents) {
            const row = document.createElement('div');
            row.className = 'doc-row';

            const title = document.createElement('span');
            title.className = 'doc-title';
            title.textContent = d.title;

            const badge = document.createElement('span');
            badge.className = 'badge ' + d.status;
            badge.textContent = d.status;

            const ver = document.createElement('span');
            ver.textContent = 'v' + d.current_version;
            ver.style.color = '#94a3b8';

            row.append(title, badge, ver);
            root.append(row);
        }
    } catch (err) {
        root.textContent = err.message;
    }
}

const uploadForm = $('uploadForm');
if (uploadForm) {
    uploadForm.onsubmit = async e => {
        e.preventDefault();
        const f = $('file').files[0];
        if (!f) return;

        if (f.size > 10 * 1024 * 1024) {
            toast('Maximum file size is 10 MB');
            return;
        }

        const fd = new FormData();
        fd.append('file', f);
        fd.append('tenant_id', ($('tenantInput').value.trim() || 'demo'));
        fd.append('owner_sub', ($('ownerSub').value.trim() || 'alice'));

        try {
            toast('Uploading...');
            const res = await fetch('/upload', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok) throw Error(data.detail || 'Upload failed');
            toast('Uploaded. Ingesting document ' + data.document_id.slice(0, 8) + '...');
            $('uploadForm').reset();
            $('ownerSub').value = 'alice';
            loadDocs();

            let polls = 0;
            const poll = setInterval(() => {
                loadDocs();
                polls += 1;
                if (polls >= 10) clearInterval(poll);
            }, 3000);
        } catch (err) {
            toast(err.message);
        }
    };
}

// ── Approvals — full workflow ─────────────────────
async function loadApprovals() {
    const root = $('approvalList');
    const summary = $('approvalSummary');
    if (!root) return;

    root.textContent = 'Loading approvals...';
    if (summary) summary.textContent = '';

    try {
        const tenant = ($('tenantInput').value.trim() || 'demo');
        const [pendingRes, allRes] = await Promise.all([
            api(`/api/approvals?tenant_id=${encodeURIComponent(tenant)}`),
            api(`/api/approvals/all?tenant_id=${encodeURIComponent(tenant)}&limit=20`),
        ]);

        const pending = pendingRes.approvals || [];
        const all = allRes.approvals || [];

        if (summary) summary.textContent = `${pending.length} pending · ${all.length} recent`;

        root.replaceChildren();

        if (all.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'approval-card';
            empty.innerHTML =
                '<strong>No approval requests yet</strong>' +
                '<p>Toggle Safe mode in the Chat tab and ask a sensitive question ' +
                '(e.g. "what is the salary for level 5") to create one.</p>';
            root.append(empty);
            return;
        }

        for (const a of all) {
            const card = document.createElement('article');
            card.className = 'approval-card';

            const header = document.createElement('div');
            header.className = 'approval-header';

            const q = document.createElement('strong');
            q.textContent = a.question;
            q.style.display = 'block';
            q.style.marginBottom = '8px';

            const badge = document.createElement('span');
            badge.className = 'approval-status ' + a.status;
            badge.textContent = a.status;

            header.append(q, badge);
            card.append(header);

            const meta = document.createElement('p');
            meta.style.fontSize = '12px';
            meta.style.color = '#94a3b8';
            meta.style.marginTop = '8px';
            meta.textContent = `Requested by ${a.requester} · ${new Date(a.created_at).toLocaleString()}`;
            card.append(meta);

            if (a.status === 'pending') {
                const actions = document.createElement('div');
                actions.style.display = 'flex';
                actions.style.gap = '10px';
                actions.style.marginTop = '12px';

                const approveBtn = document.createElement('button');
                approveBtn.textContent = 'Approve';
                approveBtn.className = 'approve-button';
                approveBtn.style.background = '#256e59';
                approveBtn.style.color = 'white';
                approveBtn.onclick = () => decideApproval(a.id, true, approveBtn);

                const rejectBtn = document.createElement('button');
                rejectBtn.textContent = 'Reject';
                rejectBtn.className = 'reject-button';
                rejectBtn.style.background = '#753443';
                rejectBtn.style.color = 'white';
                rejectBtn.onclick = () => decideApproval(a.id, false, rejectBtn);

                actions.append(approveBtn, rejectBtn);
                card.append(actions);
            } else if (a.reviewed_at) {
                const info = document.createElement('p');
                info.style.fontSize = '12px';
                info.style.color = '#94a3b8';
                info.style.marginTop = '8px';
                info.textContent = `Reviewed by ${a.reviewer || 'unknown'} · ${new Date(a.reviewed_at).toLocaleString()}`;
                card.append(info);
            }

            root.append(card);
        }
    } catch (err) {
        root.textContent = `Error loading approvals: ${err.message}`;
    }
}

async function decideApproval(approvalId, approve, button) {
    const verb = approve ? 'approve' : 'reject';
    if (!confirm(`Are you sure you want to ${verb} this request?`)) return;

    button.disabled = true;

    try {
        const res = await api(`/api/approvals/${approvalId}/decision`, {
            method: 'POST',
            body: JSON.stringify({
                reviewer: 'admin-demo',
                approve: approve,
                note: null,
            }),
        });
        toast(`Approval ${res.status}`);
        await loadApprovals();
    } catch (err) {
        toast(err.message);
        button.disabled = false;
    }
}

// ── Metrics ───────────────────────────────────────
async function loadMetrics() {
    const m = $('metrics');
    const g = $('guardrailMetrics');
    if (!m) return;

    m.textContent = 'Loading...';
    if (g) g.textContent = 'Loading...';

    try {
        const tenant = ($('tenantInput').value.trim() || 'demo');
        const [docs, apps] = await Promise.all([
            api(`/documents/${encodeURIComponent(tenant)}`),
            api(`/api/approvals/all?tenant_id=${encodeURIComponent(tenant)}&limit=500`),
        ]);

        const counts = { pending: 0, processing: 0, ready: 0, failed: 0 };
        for (const d of docs.documents || []) {
            counts[d.status] = (counts[d.status] || 0) + 1;
        }

        const appCounts = { pending: 0, approved: 0, rejected: 0, cancelled: 0 };
        for (const a of apps.approvals || []) {
            appCounts[a.status] = (appCounts[a.status] || 0) + 1;
        }

        m.textContent = JSON.stringify({
            tenant: tenant,
            total_documents: docs.count || 0,
            documents_by_status: counts,
        }, null, 2);

        if (g) {
            g.textContent = JSON.stringify({
                total_approvals: apps.count || 0,
                approvals_by_status: appCounts,
            }, null, 2);
        }
    } catch (err) {
        m.textContent = err.message;
        if (g) g.textContent = '';
    }
}