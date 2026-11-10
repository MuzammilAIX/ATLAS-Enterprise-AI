# ATLAS — Enterprise AI

**A multi-tenant enterprise RAG platform** for document intelligence, providing context-aware answers with source citations, secure data isolation, and human-in-the-loop workflows.

## Features

- **Document Ingestion:** Upload and process PDF, DOCX, TXT, and Markdown files.
- **Retrieval-Augmented Generation:** Semantic retrieval with LLM-generated answers and citations.
- **Multi-Tenant Architecture:** Tenant-scoped data isolation across PostgreSQL and Pinecone.
- **AI Guardrails:** Input validation, prompt injection detection, and evidence sanitization.
- **Human-in-the-Loop:** Approval workflows for sensitive queries.
- **Conversation Memory:** Persistent conversation history.
- **Audit Logging:** Track application activities and security events.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Python |
| Frontend | HTML, CSS, JavaScript |
| Database | PostgreSQL (Neon) |
| Vector Database | Pinecone |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Object Storage | AWS S3 |
| AI Workflows | LangGraph |

## Architecture

```text
                 DOCUMENT INGESTION
                         |
                  Document Upload
                         |
                       AWS S3
                         |
                 Extract & Chunk
                         |
                     Embeddings
                         |
                      Pinecone
                         |
                    Indexed Data

                    RAG PIPELINE
                         |
                    User Query
                         |
                   Input Validation
                         |
                  Semantic Retrieval
                         |
                  Evidence Processing
                         |
                    OpenAI LLM
                         |
                 Cited Response
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/atlas-enterprise-ai.git
cd atlas-enterprise-ai
```

### 2. Create the environment

```bash
conda create -n atlas python=3.12 -y
conda activate atlas
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file using `.env.example` and configure:

```env
DATABASE_URL=
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
S3_BUCKET=
```

### 4. Initialize and verify

```bash
python scripts/init_db.py
python -m scripts.smoke_test
```

### 5. Start the application

```bash
uvicorn app.main:app --reload --port 8000
```

Access the application:

- **Frontend:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload documents |
| POST | `/query` | Submit RAG queries |
| GET | `/documents/{tenant_id}` | Retrieve documents |
| GET | `/documents/status/{doc_id}` | Check ingestion status |
| GET | `/api/approvals` | View pending approvals |
| POST | `/api/approvals/{id}/decision` | Process approvals |
| POST | `/feedback` | Submit feedback |
| GET | `/health` | Health check |

## Project Structure

```text
atlas-enterprise-ai/
├── app/
│   ├── ingest/          # Document extraction and chunking
│   ├── static/          # Frontend assets
│   ├── templates/       # HTML templates
│   ├── main.py          # FastAPI application
│   ├── rag.py           # RAG pipeline
│   ├── graph.py         # LangGraph workflows
│   ├── guardrails.py    # Input and evidence validation
│   ├── vector.py        # Pinecone integration
│   ├── db.py            # PostgreSQL operations
│   └── ...
├── worker/              # Document ingestion worker
├── infra/               # Database schema
├── scripts/             # Initialization and smoke tests
├── test/                # Automated tests
├── requirements.txt
├── .env.example
└── README.md
```

## Security

- Tenant-scoped retrieval and database queries.
- Input validation and prompt injection detection.
- Retrieved evidence sanitization.
- Environment-based secret management.
- Audit logging for application activities.
- Human approval for sensitive operations.

## Testing

Run the automated test suite:

```bash
pytest -v
```

Integration tests cover database, storage, vector search, RAG workflows, guardrails, and API endpoints. External service credentials are required for integration tests.

## Deployment

Supports Docker-based deployment and cloud hosting platforms.

```bash
docker build -t atlas .
docker run -p 8000:8000 --env-file .env atlas
```

## License

MIT License. See [LICENSE](LICENSE).
