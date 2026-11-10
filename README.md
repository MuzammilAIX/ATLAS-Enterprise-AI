# ATLAS — Enterprise AI

**An enterprise-grade, multi-tenant RAG platform for secure document intelligence and grounded AI responses.**

ATLAS enables organizations to upload, retrieve, and query private documents through an AI-powered knowledge assistant with source citations, tenant isolation, and controlled response generation.

## Key Features

- **Multi-Tenant RAG:** Isolated document retrieval for multiple organizations.
- **Document Intelligence:** Upload and process PDF, DOCX, TXT, and Markdown files.
- **Grounded Responses:** Generate answers using retrieved document context with citations.
- **Hybrid Enterprise Architecture:** Pinecone vector search and PostgreSQL persistence.
- **AI Guardrails:** Input validation, prompt injection detection, and relevance filtering.
- **Human-in-the-Loop:** Approval workflows for sensitive operations.
- **Conversation Memory:** Context-aware conversation history.
- **Audit & Observability:** Track operations and monitor application behavior.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI / LLM | OpenAI API |
| Embeddings | Sentence Transformers (BGE) |
| Vector Database | Pinecone |
| Database | PostgreSQL (Neon) |
| Cloud Storage | AWS S3 |
| AI Workflow | LangGraph |
| Deployment | Docker |

## Architecture

```text
Documents
    |
    v
Document Ingestion
    |
    v
Chunking & Embeddings
    |
    v
Pinecone Vector Database
    |
    v
User Query
    |
    v
Retrieval & Relevance Filtering
    |
    v
LLM + Guardrails
    |
    v
Grounded Answer + Citations
```

## Getting Started

**Prerequisites**
- Python 3.12+
- Docker
- OpenAI API key
- Pinecone account
- PostgreSQL database
- AWS S3 bucket

**1. Clone the repository**

```bash
git clone https://github.com/MuzammilAIX/ATLAS-Enterprise-AI.git
cd ATLAS-Enterprise-AI
```

**2. Configure environment**

```bash
cp .env.example .env
```

Add your API keys, database credentials, and service configurations to `.env`.

**3. Build Docker image**

```bash
docker build -t atlas .
```

**4. Run the application**

```bash
docker run -p 8000:8000 --env-file .env atlas
```

**5. Access the application**

- Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Structure

```text
ATLAS-Enterprise-AI/
├── app/              # FastAPI application and AI modules
├── worker/           # Document ingestion pipeline
├── infra/            # Database schema
├── scripts/          # Initialization utilities
├── test/             # Automated tests
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Security

- Tenant-scoped data access.
- Environment-based secret management.
- Prompt injection detection.
- Retrieval-based response validation.
- Auditable application operations.

---

**Author:** [MuzammilAIX](https://github.com/MuzammilAIX)

**Project:** ATLAS — Enterprise AI
