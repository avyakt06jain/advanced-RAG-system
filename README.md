# Advanced RAG System for Insurance Policy Analysis

> A production-ready Retrieval-Augmented Generation (RAG) system built for HackRX 6.0 — designed to ingest complex insurance policy PDFs and answer natural language questions with verifiable citations.

## Demo Video

<!-- TODO: Record and embed demo video here -->
<!-- Example: -->
<!-- [![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) -->

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [Testing](#testing)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview

Insurance policy documents are complex — dense legal language, multi-column layouts, nested tables, and distributed logic across sections. This system solves that challenge by building a multi-stage RAG pipeline that:

1. **Parses** PDFs with high fidelity using PyMuPDF
2. **Chunks** content intelligently based on document structure (not arbitrary length)
3. **Indexes** using FAISS for fast semantic search
4. **Retrieves** via hybrid search (semantic + keyword) with cross-encoder re-ranking
5. **Generates** grounded answers using Google Gemini with full source citations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Offline Indexing Pipeline               │
│                                                         │
│  PDF → PyMuPDF Parser → Structural Chunking → Embedding │
│                                              → FAISS     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Online Inference Pipeline               │
│                                                         │
│  User Query → Hybrid Search (BM25 + Semantic)           │
│             → Cross-Encoder Re-ranking                   │
│             → Context Augmentation → Gemini LLM → Answer │
└─────────────────────────────────────────────────────────┘
```

## Features

- **High-fidelity PDF parsing** with PyMuPDF — preserves structure, font metadata, and layout
- **Intelligent structural chunking** — respects document hierarchy (sections, subsections, lists)
- **Header/footer suppression** — automatically detects and removes repeating content
- **Hybrid retrieval** — combines BM25 keyword search with semantic vector search via Reciprocal Rank Fusion
- **Cross-encoder re-ranking** — uses `ms-marco-MiniLM-L-6-v2` for precision re-ranking
- **Grounded generation** — Gemini 2.0 Flash with strict context-only instructions
- **Source citations** — every answer includes document name, page number, and section
- **Document caching** — processed documents are cached by hash to avoid re-ingestion
- **In-memory index caching** — FAISS indexes and chunks cached in memory for fast repeated queries
- **REST API** — FastAPI with Bearer token authentication
- **Docker-ready** — fully containerized with one-command deployment

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| PDF Parsing      | PyMuPDF (Fitz)                      |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store     | FAISS (CPU)                         |
| Keyword Search   | rank-bm25                           |
| Re-ranking       | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM              | Google Gemini 2.0 Flash             |
| API Framework    | FastAPI + Uvicorn                   |
| Containerization | Docker                              |

## Project Structure

```
advanced-RAG-system/
├── app.py                              # FastAPI application & API endpoints
├── ingestion_pipeline/
│   ├── __init__.py
│   └── ingestionPipeline.py            # PDF parsing, chunking, embedding, FAISS indexing
├── inference_pipeline/
│   ├── __init__.py
│   └── inferencePipeline.py            # Hybrid search, re-ranking, answer generation
├── testapi.py                          # Test script with sample queries
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container configuration
├── .env.example                        # Environment variables template
├── .gitignore
├── hackrx/                             # Hackathon reference materials
└── README.md                           # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- A Google Gemini API key ([get one here](https://ai.google.dev/))
- (Optional) Docker & Docker Compose

### Local Setup

1. **Clone and create virtual environment:**

```bash
cd advanced-RAG-system
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
API_KEY=your_secret_api_key
GEMINI_API_KEY=your_gemini_api_key
```

4. **Start the server:**

```bash
python app.py
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API docs: `http://127.0.0.1:8000/docs`

### Docker Setup

1. **Build the image:**

```bash
docker build -t advanced-rag-system .
```

2. **Run with environment variables:**

```bash
docker run -p 8000:8000 \
  -e API_KEY=your_secret_api_key \
  -e GEMINI_API_KEY=your_gemini_api_key \
  advanced-rag-system
```

## API Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "Healthy",
  "timestamp": "2026-04-04T12:00:00.000000"
}
```

### Process Document & Answer Queries

```
POST /hackrx/run
Authorization: Bearer <API_KEY>
```

**Request Body:**
```json
{
  "document": "https://example.com/policy.pdf",
  "queries": [
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The waiting period for pre-existing diseases is 24 months from the policy inception date [Source: policy.pdf, Page 15, Section: Waiting Periods].",
    "Yes, maternity expenses are covered subject to a 9-month waiting period and a sub-limit of ₹50,000 per delivery [Source: policy.pdf, Page 22, Section: Maternity Benefits]."
  ]
}
```

## Testing

Run the included test script to verify the API:

```bash
# Set your API key first
export API_KEY=your_api_key_here

python testapi.py
```

Or test via Swagger UI at `http://127.0.0.1:8000/docs`.

## How It Works

### Ingestion Pipeline

1. **PDF Parsing** — PyMuPDF extracts text with word-level metadata (position, font size, weight)
2. **Header/Footer Detection** — Repeating content across pages is identified and suppressed
3. **Structural Chunking** — Content is split by logical document structure (sections, lists, paragraphs), not fixed character counts
4. **List Merging** — Related list items are merged with their parent context
5. **Metadata Enrichment** — Each chunk gets source document, page number, and section header
6. **Embedding** — Chunks are vectorized using `all-MiniLM-L6-v2`
7. **FAISS Indexing** — Vectors are stored in a FAISS L2 index, chunks saved as JSON

### Inference Pipeline

1. **Hybrid Retrieval** — Both BM25 (keyword) and semantic (FAISS) search run in parallel
2. **Reciprocal Rank Fusion** — Results from both searches are merged using RRF
3. **Cross-Encoder Re-ranking** — Top fused results are re-scored for maximum relevance
4. **Context Augmentation** — Top 5 chunks are formatted with source metadata
5. **Grounded Generation** — Gemini generates answers strictly from provided context with citations

## Future Enhancements

- Domain-specific embedding model fine-tuning (e.g., LEGAL-BERT)
- Advanced RAG techniques: Self-RAG, Corrective RAG (CRAG)
- Multi-document comparison (agentic architecture)
- Streaming responses for lower perceived latency
- Conversational memory for follow-up questions
- Table-specific extraction with PDFPlumber
- Metadata filtering on FAISS queries

## License

This project was built for HackRX 6.0.
