# Recipe RAG: Hybrid Retrieval + Dialogue State Tracking

This repository implements a Retrieval-Augmented Generation (RAG) system for recipe question answering.
It supports:
- Recipe preprocessing into a normalized JSONL schema
- Section-aware chunking (ingredients / notes / steps / info)
- Hybrid retrieval (BM25 + dense embeddings with FAISS)
- Reciprocal Rank Fusion (RRF) to merge retrieval results
- Dialogue State Tracking (DST) to rewrite follow-up queries into standalone queries
- Metadata/tag-based filtering prior to generation

## Repository structure
├── notebook/data/                  # Input recipes as .txt files (raw)
├── notebook/schema/                # Generated artifacts (documents.jsonl, chunks.jsonl, FAISS index)
├── notebook/RAG.ipynb                # End-to-end notebook (preprocess → index → interactive QA)
├── README.md
└── evaluation          # evaluation setup and results.

## Requirements
Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Select this environment for the kernel in your notebook.

There is and error analysis in the evaluation folder.
