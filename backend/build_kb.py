# backend/build_kb.py
import os
import json
from pathlib import Path
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "backend" / "kb"
KB_DIR.mkdir(parents=True, exist_ok=True)

DOCX_PATH = DATA_DIR / "RMW Training Data 3.docx"  # put the file here
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = KB_DIR / "faiss_index.bin"
META_PATH = KB_DIR / "metadata.json"

def read_docx(path):
    doc = Document(path)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs

def chunk_text(paragraphs, max_tokens=200):
    # Simple chunk: join paragraphs until ~max length (characters heuristic)
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 > 1500:  # heuristic cutoff
            chunks.append(cur.strip())
            cur = p
        else:
            cur += "\n\n" + p if cur else p
    if cur:
        chunks.append(cur.strip())
    return chunks

def build_index(chunks, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (we'll L2-normalize)
    # normalize to use cosine similarity with IP
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def save_index(index, meta_chunks):
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": meta_chunks}, f, ensure_ascii=False, indent=2)
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {META_PATH}")

def main():
    assert DOCX_PATH.exists(), f"Please place your docx at: {DOCX_PATH}"
    print("Reading DOCX...")
    paragraphs = read_docx(DOCX_PATH)
    print(f"Extracted {len(paragraphs)} paragraphs.")
    chunks = chunk_text(paragraphs)
    print(f"Created {len(chunks)} chunks.")
    print("Building index (this will download the model if needed)...")
    index, embeddings = build_index(chunks)
    save_index(index, chunks)
    print("Done.")

if __name__ == "__main__":
    main()