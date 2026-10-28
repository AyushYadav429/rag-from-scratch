# RAG From Scratch

A hands-on implementation of Retrieval-Augmented Generation (RAG) using Python, SentenceTransformers, FAISS, and Gemma 2B via Ollama.

## Overview

This project demonstrates how a RAG pipeline retrieves relevant information from a knowledge base and uses a large language model to generate an answer grounded in the retrieved context.

The pipeline is implemented without a high-level RAG framework, making the underlying architecture easier to understand.

## Architecture

1. **Chunking:** Splits source text into smaller passages.
2. **Embeddings:** Converts text into 384-dimensional vectors using `all-MiniLM-L6-v2`.
3. **Vector indexing:** Stores embeddings in a FAISS `IndexFlatL2` index.
4. **Retrieval:** Finds the top-k nearest document vectors for a user query.
5. **Generation:** Passes the retrieved context and question to Gemma 2B through Ollama.

## Tech Stack

- Python
- SentenceTransformers
- FAISS
- NumPy
- Ollama
- Gemma 2B

## Prerequisites

- Python
- Git
- Ollama installed locally
- The `gemma:2b` model downloaded in Ollama

## Setup

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd rag-from-scratch
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the language model:

```powershell
ollama pull gemma:2b
```

## Run the project

Run the integrated RAG application:

```powershell
python main.py
```

Enter a question when prompted.

## Current Limitations

- Uses a small, hardcoded knowledge base.
- Recreates the embeddings and FAISS index each time the application starts.
- Uses basic chunking without overlap.
- Uses vector retrieval only; BM25 hybrid retrieval is not implemented.
- Does not yet include automated retrieval evaluation or source citations.

## Planned Improvements

- Ingest real TXT and PDF documents.
- Persist the FAISS index and associated document metadata.
- Implement overlapping chunking.
- Add BM25 hybrid retrieval with Reciprocal Rank Fusion.
- Evaluate retrieval quality and answer faithfulness.
- Expose the pipeline through FastAPI.

## Learning Objective

The goal is to understand the internal mechanics of RAG, from document processing and vector retrieval to LLM-based answer generation.