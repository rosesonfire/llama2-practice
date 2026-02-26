# LLaMA 2 RAG System

A Retrieval-Augmented Generation (RAG) system powered by LLaMA 2, combining local LLM inference with semantic search for question-answering.

## Overview

This project implements a RAG pipeline that:
- **Retrieves** relevant document chunks using FAISS vector search
- **Augments** prompts with retrieved context
- **Generates** answers using LLaMA 2 running locally via llama-cpp

The system uses sentence transformers for embeddings and supports both API and CLI interfaces.

## Project Structure

```
├── app.py              # FastAPI server with /chat endpoint
├── rag.py              # CLI interface for testing RAG
├── generator.py        # LLaMA 2 response generation
├── retriever.py        # FAISS-based document retrieval
├── ingest.py           # Build vector index from documents
├── models/             # Model files
│   ├── llama-2-7b-chat.Q4_K_M.gguf
│   └── all-MiniLM-L6-v2/          # Sentence transformer for embeddings
├── vendors/            # llama.cpp binary and libraries
└── llama2-practice.faiss          # Built FAISS index
```

## Features

- **Local LLM**: Runs LLaMA 2 7B locally without external APIs
- **Vector Search**: FAISS-based semantic retrieval of relevant documents
- **Context-Aware**: Generates answers using retrieved context
- **FastAPI**: REST API for integration with other applications

## Setup

### Prerequisites
- Python 3.10+

### Tested On
- **CPU**: AMD Ryzen 9 9950X (16 cores / 32 threads)
- **RAM**: 64 GB
- **Storage**: 2 TB
- **OS**: Ubuntu 24.04 (64-bit)
- **GPU**: None (CPU-only system with AVX/AVX2/AVX512 support)

### Installation

1. **Install Poetry**:
```bash
pip install setuptools poetry
```

1. **Create a Python 3.10 virtual environment** and activate it.

1. **Install dependencies with Poetry** (you can define a custom virtualenv name):
```bash
# set name before creating the environment
export POETRY_VIRTUALENVS_NAME="llama2-practice-poetry"
poetry install
```
   After installation you must execute code inside the poetry environment. Either:
   ```bash
   poetry env activate        # activate the environment (Poetry 2.x)
   poetry run python rag.py   # run a script directly
   ```
   
   (or install the `shell` plugin if you prefer the old `poetry shell` command)


1. **Populate gitignored files** (after cloning):

The following directories/files are gitignored and must be populated:

- **`models/llama-2-7b-chat.Q4_K_M.gguf`** (~3.5GB)
  - Download from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf)
  - Place in `models/` directory

- **`models/all-MiniLM-L6-v2/`**
  - `hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2`

- **`vendors/llama-b7999/`** (precompiled llama.cpp binaries)
  - Obtain from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)

- **`llama2-practice.faiss` and `metadata.pkl`** (generated)
  - Build the index after populating models:
  ```bash
  python ingest.py
  ```

## Usage

### API Server
Run inside the Poetry environment (or prefix with `poetry run`):
```bash
poetry run uvicorn app:app --reload
```

(installing a global `uvicorn` with apt is not recommended; the project dependency is managed by Poetry)

Send queries to the `/chat` endpoint:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Redis?"}'
```

### CLI Testing
```bash
python rag.py
```

Query the RAG system directly and print the response.

## Models

- **LLM**: LLaMA 2 7B Chat (quantized to Q4_K_M format)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim sentence transformers)
- **Vector DB**: FAISS IndexFlatL2

## How It Works

1. **Ingestion**: Documents are chunked and embedded using sentence transformers
2. **Storage**: Embeddings stored in FAISS index, chunks cached in pickle
3. **Retrieval**: User query embedded and matched against k nearest chunks
4. **Generation**: Retrieved context + query prompt sent to LLaMA 2
5. **Output**: Generated answer returned to user

## Configuration

Adjust in source files:
- `generator.py`: LLaMA 2 parameters (n_threads, n_ctx, max_tokens)
- `retriever.py`: Number of retrieved chunks (k=3)
- `ingest.py`: Chunk size and overlap (500 chars, 50 overlap)
