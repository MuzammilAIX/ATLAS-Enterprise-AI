// ATLAS single-page UI — wired to /query, /upload, /documents, /api/approvals.

const $ = id => document.getElementById(id);

let conversationId = sessionStorage.getItem('atlas_conversation_id') || crypto.randomUUID();
sessionStorage.setItem('atlas_conversation_id', conversationId);

$('newChat').onclick = () => {
    conversationId = crypto.randomUUID();
    sessionStorage.setItem('atlas_conversation_id', conversationId);
    $('messages').replaceChildren();
    $('heroBlock').classList.remove('hidden');
    toast('Started a new conversation');
};

const toast = msg => {
    const el = $('toast');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
};

const api = async (path, options = {}) => {
    const res = await fetch(path, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    if (!res.ok) throw Error(data.detail || `HTTP ${res.status}`);
    return data;
};

function view(name) {
    ['chat', 'docs', 'approval', 'admin'].forEach(x => {
        $(x + 'View').classList.toggle('hidden', x !== name);
        $(x + 'Tab').classList.toggle('active', x === name);
    });
    $('heading').textContent = {
        chat: "Ask your organization's knowledge",
        docs: 'Document management',
        approval: 'Approval requests',
        admin: 'Operational overview',
    }[name];
    if (name === 'docs') loadDocs();
    if (name === 'admin') loadMetrics();
    if (name === 'approval') loadApprovals();
}

$('chatTab').onclick = () => view('chat');
$('docsTab').onclick = () => view('docs');
$('adminTab').onclick = () => view('admin');
$('approvalTab').onclick = () => view('approval');
$('refreshApprovals').onclick = loadApprovals;

function addMessage(content, kind, sources = [], extra = null) {
    const el = document.createElement('div');
    el.className = 'message ' + kind;
    el.textContent = content;

    if (sources.length) {
        const wrap = document.createElement('div');
        wrap.className = 'sources';
        sources.forEach((s, i) => {
            const d = document.createElement('details');
            const title = document.createElement('summary');
            const score = (s.score ?? 0).toFixed(3);
            const docId = s.metadata?.document_id || 'unknown';
            title.textContent = `[${i + 1}] doc=${docId} · score=${score}`;
            const excerpt = document.createElement('p');
            excerpt.textContent = (s.metadata?.text || '').slice(0, 400);
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

    $('messages').append(el);
    $('messages').scrollTo({ top: $('messages').scrollHeight, behavior: 'smooth' });
}

$('chatForm').onsubmit = async e => {
    e.preventDefault();
    const q = $('question').value.trim();
    if (!q) return;

    addMessage(q, 'user');
    $('question').value = '';
    $('heroBlock').classList.add('hidden');

    const loading = document.createElement('div');
    loading.className = 'message assistant';
    loading.textContent = 'Searching approved documents...';
    $('messages').append(loading);
    $('messages').scrollTo({ top: $('messages').scrollHeight, behavior: 'smooth' });

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
                'Reason: ' + r.approval_reason,
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

document.querySelectorAll('[data-q]').forEach(b => {
    b.onclick = () => {
        $('question').value = b.dataset.q;
        $('question').focus();
    };
});

// ── Documents ──────────────────────────────────────

async function loadDocs() {
    const root = $('docs');
    root.textContent = 'Loading...';
    try {
        const tenant = $('tenantInput').value.trim() || 'demo';
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
            ver.textContent = `v${d.current_version}`;
            ver.style.color = '#94a3b8';

            row.append(title, badge, ver);
            root.append(row);
        }
    } catch (err) {
        root.textContent = err.message;
    }
}

$('uploadForm').onsubmit = async e => {
    e.preventDefault();
    const f = $('file').files[0];
    if (!f) return;

    if (f.size > 10 * 1024 * 1024) {
        toast('Maximum file size is 10 MB');
        return;
    }

    const fd = new FormData();
    fd.append('file', f);
    fd.append('tenant_id', $('tenantInput').value.trim() || 'demo');
    fd.append('owner_sub', $('ownerSub').value.trim() || 'alice');

    try {
        toast('Uploading...');
        const res = await fetch('/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw Error(data.detail || 'Upload failed');
        toast(`Uploaded. Ingesting document ${data.document_id.slice(0, 8)}...`);
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

// ── Approvals — FULL WORKFLOW ─────────────────────

async function loadApprovals() {
    const root = $('approvalList');
    const summary = $('approvalSummary');

    root.textContent = 'Loading approvals...';
    summary.textContent = '';

    try {
        const tenant = $('tenantInput').value.trim() || 'demo';
        const [pendingRes, allRes] = await Promise.all([
            api(`/api/approvals?tenant_id=${encodeURIComponent(tenant)}`),
            api(`/api/approvals/all?tenant_id=${encodeURIComponent(tenant)}&limit=20`),
        ]);

        const pending = pendingRes.approvals || [];
        const all = allRes.approvals || [];

        summary.textContent = `${pending.length} pending · ${all.length} recent`;

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

// ── Metrics ────────────────────────────────────────

async function loadMetrics() {
    const m = $('metrics');
    const g = $('guardrailMetrics');

    m.textContent = 'Loading...';
    g.textContent = 'Loading...';

    try {
        const tenant = $('tenantInput').value.trim() || 'demo';
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
            tenant,
            total_documents: docs.count || 0,
            documents_by_status: counts,
        }, null, 2);

        g.textContent = JSON.stringify({
            total_approvals: apps.count || 0,
            approvals_by_status: appCounts,
        }, null, 2);
    } catch (err) {
        m.textContent = err.message;
        g.textContent = '';
    }
}