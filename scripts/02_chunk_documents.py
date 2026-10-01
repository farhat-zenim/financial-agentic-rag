"""
scripts/02_chunk_documents.py
──────────────────────────────
Entry-point script for the chunking stage.  Run from the project root:

    python scripts/02_chunk_documents.py

What it does
------------
1. Reads every *_clean.md from parsed_docs/markdown_clean/
2. Loads the matching Docling JSON from parsed_docs/json/ for page-number
   resolution (gracefully skips if a JSON is absent).
3. Calls src.ingestion.chunker.run_chunking_pipeline() which:
     - Tracks active ## section headings across blocks
     - Keeps table blocks whole (never split mid-row)
     - Splits long prose with SentenceSplitter (512-token window, 64-token overlap)
     - Enriches every chunk with: doc_id, company, filing_type, fiscal_year,
       fiscal_quarter, section, page_no, chunk_type, word_count, char_count
4. Writes parsed_docs/chunks/<doc_id>_chunks.jsonl  (one chunk per line)
5. Updates parsed_docs/chunk_manifest.json after every document

Output layout
-------------
parsed_docs/
├── chunks/
│   ├── 3M_2015_10K_chunks.jsonl
│   ├── 3M_2016_10K_chunks.jsonl
│   └── ... (one file per document)
└── chunk_manifest.json

Each line in a .jsonl file is a JSON object:
{
  "chunk_id":       "c8f59f468b9b",   # MD5(doc_id+index)[:12] — stable for upserts
  "chunk_index":    16,
  "doc_id":         "3M_2015_10K",
  "company":        "3M",
  "filing_type":    "10-K",
  "fiscal_year":    "2015",
  "fiscal_quarter": null,
  "source":         "SEC EDGAR",
  "section":        "Item 1A. Risk Factor s.",
  "page_no":        9,
  "chunk_type":     "prose",           # "prose" | "table"
  "word_count":     87,
  "char_count":     512,
  "text":           "..."
}

Prerequisites
-------------
    pip install llama-index-core
    # (already in your requirements.txt via the RAG blueprint)
"""

import os
import sys

# Make sure 'src' is importable when the script is run from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import run_chunking_pipeline

if __name__ == "__main__":
    run_chunking_pipeline()
