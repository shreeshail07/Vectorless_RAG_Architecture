# Enterprise Insurance Claims Intelligence Platform
### Vectorless Retrieval-Augmented Generation (RAG)

An explainable, enterprise-oriented Retrieval-Augmented Generation platform designed to help insurance professionals retrieve and analyze claims information using lexical search, exact identifier matching, and structured metadata filtering — without relying on embeddings or vector databases.

## 1. Project Overview

Insurance claims processing involves large volumes of structured and unstructured information, including:

- Claim records
- Policy information
- Investigation reports
- Claim notes
- Supporting PDF and DOCX documents
- Assessment and settlement information

Finding relevant information across these sources can be time-consuming.

This project aims to build an intelligent claims assistant that retrieves relevant information, provides contextual answers, and maintains traceability to the underlying documents.

## 2. Core Architecture

The retrieval engine follows a Vectorless RAG approach.

Instead of generating embeddings and searching a vector database, it combines:

- Exact claim and policy identifier matching
- Structured metadata filtering
- BM25 lexical ranking
- Result fusion and relevance scoring
- Document-level source tracking

## 3. Current Implemented Capabilities

- Multi-format document ingestion: CSV, TXT, PDF and DOCX.
- Claim and policy identifier extraction.
- Structured metadata filtering.
- BM25 lexical retrieval.
- Exact-match retrieval.
- Retrieval result fusion and ranking.
- Document caching.
- Coordinated document and BM25 cache refresh.
- Fault isolation during document ingestion.
- Automated tests for ingestion and retrieval reliability.

## 4. Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Package management | UV |
| Data processing | Pandas |
| Retrieval | BM25, custom Python retrieval engine |
| PDF processing | pypdf |
| DOCX processing | python-docx |
| Frontend | Streamlit |
| Backend | FastAPI (development roadmap) |
| Testing | Pytest |
| Version control | Git and GitHub |

## 5. Target Enterprise Architecture

The following components are part of the planned architecture, not all currently implemented.

- Oracle Database — existing insurance claims source system.
- PostgreSQL — application and audit data.
- Amazon S3 — document storage.
- Amazon SQS — asynchronous ingestion.
- Amazon OpenSearch — BM25 lexical retrieval.
- Amazon Bedrock — LLM integration.
- Docker and Amazon ECS — deployment.
- Terraform — infrastructure as code.
- GitHub Actions — CI/CD.
- AWS CloudWatch — monitoring and logging.

## 6. Security and Governance Goals

The production design will incorporate:

- Role-based access control.
- Claim-level authorization.
- Secure handling of customer information.
- Encryption at rest and in transit.
- Retrieval and response audit trails.
- Source citations and provenance.
- Prompt-injection protection.
- Human review for claim decisions.

The AI assistant is intended to support claim investigation and information retrieval, not independently approve or reject insurance claims.

## 7. Development Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1 | Core RAG architecture and ingestion reliability | In progress |
| M2 | FastAPI backend | Planned |
| M3 | Oracle claims-system integration | Planned |
| M4 | S3 document ingestion | Planned |
| M5 | OpenSearch BM25 integration | Planned |
| M6 | LLM integration and claims reporting | Planned |
| M7 | Docker and local deployment | Planned |
| M8 | AWS deployment | Planned |
| M9 | CI/CD, security and monitoring | Planned |

## 8. Local Development

Clone the repository:

```bash
git clone https://github.com/shreeshail07/Vectorless_RAG_Architecture.git
cd Vectorless_RAG_Architecture
```

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -v
```

## 9. Development Branch

Active development takes place on:

```bash
feature/production-architecture
```

The `main` branch is maintained separately from ongoing development.

## 10. Project Objective

Build a production-oriented, explainable insurance claims intelligence platform that demonstrates practical engineering across Python, retrieval systems, databases, APIs, cloud infrastructure, testing, security, and DevOps.

**Core principle: Accurate retrieval, explainable results, secure access, and traceable AI assistance.**