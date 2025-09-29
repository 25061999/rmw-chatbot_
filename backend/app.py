# backend/app.py
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import faiss
import re
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
INDEX_PATH = KB_DIR / "faiss_index.bin"
META_PATH = KB_DIR / "metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

app = Flask(__name__, static_folder="../static", static_url_path="/static")

# Load model + index + metadata
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Loading metadata...")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
chunks = meta["chunks"]

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

def embed_text(text):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))
    if not query:
        return jsonify({"error": "empty query"}), 400

    # Embed query and search FAISS
    q_emb = embed_text(query)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "text": chunks[idx]
        })

    if not results:
        return jsonify({
            "query": query,
            "answer": "Sorry, I couldn't find an answer.",
            "sources": []
        })

    # Combine top chunks
    combined_text = " ".join([r["text"] for r in results])
    lines = combined_text.splitlines()

    # --- Structured info ---
    structured_info = {}
    structured_info["phones"] = list(set(re.findall(r"\+?\d[\d\s-]{7,}", combined_text)))
    structured_info["emails"] = list(set(re.findall(r"[\w\.-]+@[\w\.-]+", combined_text)))
    structured_info["addresses"] = [line.strip() for line in lines if any(k in line.lower() for k in ["road", "street", "floor", "sector", "building", "park", "city", "tower"])]
    structured_info["founders_ceo"] = [line.strip() for line in lines if any(k in line.lower() for k in ["founder", "ceo"])]
    structured_info["websites"] = list(set(re.findall(r"https?://\S+", combined_text)))

    # --- Intent detection for specific fields ---
    q_lower = query.lower()
    if any(k in q_lower for k in ["phone", "contact", "call"]):
        answer = structured_info["phones"][0] if structured_info["phones"] else "Phone not found."
    elif any(k in q_lower for k in ["email", "mail"]):
        answer = structured_info["emails"][0] if structured_info["emails"] else "Email not found."
    elif any(k in q_lower for k in ["address", "location", "office", "headquarters"]):
        answer = "\n".join(structured_info["addresses"]) if structured_info["addresses"] else "Address not found."
    elif any(k in q_lower for k in ["founder", "ceo", "ceos"]):
        answer = "\n".join(structured_info["founders_ceo"]) if structured_info["founders_ceo"] else "Founder/CEO not found."
    elif any(k in q_lower for k in ["website", "url", "site"]):
        answer = structured_info["websites"][0] if structured_info["websites"] else "Website not found."
    else:
        # --- For any random question: return most relevant sentence(s) ---
        # Split combined text into sentences
        sentences = re.split(r'(?<=[.!?]) +', combined_text)
        query_words = set(re.findall(r'\w+', query.lower()))
        sentence_scores = []
        for s in sentences:
            words = set(re.findall(r'\w+', s.lower()))
            score = len(query_words & words)  # count of overlapping words
            sentence_scores.append((score, s))
        sentence_scores.sort(reverse=True)
        # Take top 2 matching sentences if score > 0
        top_sentences = [s for score, s in sentence_scores if score > 0][:2]
        answer = "\n".join(top_sentences) if top_sentences else sentences[0]  # fallback: first sentence

    return jsonify({
        "query": query,
        "answer": answer,
        "sources": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
