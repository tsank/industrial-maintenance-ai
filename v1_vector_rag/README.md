# Maintenance RAG

A production-grade Retrieval-Augmented Generation (RAG) system for industrial equipment manuals.

Built as Problem 1 of a four-problem LangChain/LangGraph learning progression.

## What it does

Answers technician questions grounded strictly in uploaded equipment manuals. Uses a multi-strategy PDF extraction cascade to handle corrupted font encodings, scanned pages, and menu diagram noise - common in real-world industrial PDFs.

## Stack

- **LangChain 0.3.7** - RAG chain via LCEL pipe syntax
- **ChromaDB 0.5.23** - vector store (runs in Docker)
- **OpenAI** - text-embedding-3-small for embeddings, gpt-4o for answers
- **PyMuPDF + pdfplumber + pytesseract** - three-strategy PDF extraction cascade

## Project structure

```
maintenance-rag/
├── docs/                  # Add your PDF manuals here
├── src/
│   ├── app.py             # CLI interface
│   ├── chain.py           # RAG chain builder
│   ├── ingest.py          # PDF ingestion pipeline
│   ├── pdf_loader.py      # Robust document extraction
│   └── retriever.py       # Vector store helper (empty placeholder)
├── docker-compose.yml     # Local ChromaDB stack
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
└── README.md
```

## Setup

1. Install system dependencies
   - `tesseract`
   - `poppler`

2. Create a Python environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the repository root with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   CHROMA_HOST=localhost
   CHROMA_PORT=8000
   ```

## Running the system

Start ChromaDB with Docker Compose:
```bash
docker compose up -d
```

Ingest manuals into the vector store:
```bash
python src/ingest.py
```

Run the interactive CLI:
```bash
python src/app.py
```

## How it works

- `src/ingest.py` loads PDFs from `docs/`, extracts text, chunks it, embeds it, and stores the vectors in ChromaDB.
- `src/pdf_loader.py` uses a three-strategy extraction cascade:
  1. PyMuPDF for native text
  2. pdfplumber for structured text extraction
  3. pytesseract OCR as a fallback for garbled or scanned pages
- `src/chain.py` builds a LangChain RAG chain and queries OpenAI with retrieved excerpts.
- `src/app.py` provides an interactive terminal Q&A loop for technician questions.

## Notes

- `docs/` and PDF files are excluded from source control.
- Do not commit proprietary or confidential manuals to this repository.
- If `.env` is not loaded automatically, verify the `load_dotenv` path in `src/chain.py` and `src/ingest.py`.

## Document source

The sample manual used for development and testing is the Atlas Copco GA5 User Manual (Document No. 2920 1461 03), sourced from ManualsLib: https://www.manualslib.com/manual/1234567/Atlas-Copco-Ga5.html

This document is used strictly for educational and research purposes. It is excluded from this repository via .gitignore (`docs/` and `*.pdf`).

To use this system with your own manuals, place any PDF in the `docs/` folder and run `python src/ingest.py`. **Do not commit proprietary or confidential documents to this repository.**
