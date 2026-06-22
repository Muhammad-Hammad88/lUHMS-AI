"""
embed.py
--------
Embedding module for LUMHS chatbot.

Two modes:
1. STANDALONE — run directly to embed entire scraped_data.json
2. FUNCTION   — import embed_pages() for use by server.py admin updates

Improvements:
- Section-aware metadata (used for targeted deletion)
- Paragraph + sentence aware chunking with overlap
- Duplicate prevention via content hashing
- Batch processing for performance
- Progress callback support for live admin panel logs
"""

import json
import re
import hashlib
from typing import Optional, Callable

import chromadb
from sentence_transformers import SentenceTransformer

from sections import get_section_for_url

# =========================
# CONFIG
# =========================
CHROMA_PATH = "./chroma_db"
DATA_FILE = "scraped_data.json"
BATCH_SIZE = 64
MAX_WORDS_PER_CHUNK = 180
OVERLAP_WORDS = 20
MIN_CHUNK_WORDS = 20

# =========================
# NOISE FILTER
# =========================
NOISE_PHRASES = [
    "gateway to a brighter future",
    "copyright",
    "all rights reserved",
    "home about contact",
    "student portal",
    "staff portal",
]

def is_noise(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in NOISE_PHRASES)

# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# SMART CHUNKING
# Splits by paragraphs first, then sentences
# Maintains overlap between chunks for context continuity
# =========================
def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS) -> list[str]:
    # Try paragraph-based splitting first
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\.\s{2,}', text) if p.strip()]

    chunks = []
    current_chunk: list[str] = []
    current_words = 0

    def flush_chunk():
        if current_chunk:
            chunk = " ".join(current_chunk)
            if len(chunk.split()) >= MIN_CHUNK_WORDS and not is_noise(chunk):
                chunks.append(chunk)

    def get_overlap_words(word_list: list[str]) -> list[str]:
        return word_list[-overlap:] if overlap and len(word_list) >= overlap else []

    for para in paragraphs:
        para_words = para.split()
        para_len = len(para_words)

        if para_len > max_words:
            # Paragraph too long — split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_words = sentence.split()
                s_len = len(s_words)

                if current_words + s_len > max_words and current_chunk:
                    flush_chunk()
                    overlap_words = get_overlap_words(current_chunk)
                    current_chunk = overlap_words + s_words
                    current_words = len(current_chunk)
                else:
                    current_chunk.extend(s_words)
                    current_words += s_len
        else:
            if current_words + para_len > max_words and current_chunk:
                flush_chunk()
                overlap_words = get_overlap_words(current_chunk)
                current_chunk = overlap_words + para_words
                current_words = len(current_chunk)
            else:
                current_chunk.extend(para_words)
                current_words += para_len

    flush_chunk()

    # Fallback — if no chunks produced, use sliding window
    if not chunks:
        words = text.split()
        for i in range(0, len(words), max_words - overlap):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) >= MIN_CHUNK_WORDS and not is_noise(chunk):
                chunks.append(chunk)

    return chunks

# =========================
# METADATA EXTRACTION
# =========================
def extract_metadata(url: str, text: str, section: Optional[str] = None) -> dict:
    url_lower = url.lower()
    text_lower = text.lower()

    # Detect most recent year mentioned
    years = re.findall(r"(20\d{2})", text_lower)
    latest_year = max([int(y) for y in years]) if years else 0

    # Use provided section or detect from URL
    if not section:
        section = get_section_for_url(url)

    # Detect document type and priority
    if url_lower.endswith(".pdf"):
        doc_type = "document"
        priority = 9
    elif "admission" in url_lower or "admission" in text_lower:
        doc_type = "admission"
        priority = 10
    elif "result" in url_lower:
        doc_type = "result"
        priority = 8
    elif "fee" in url_lower or "fee structure" in text_lower:
        doc_type = "fee"
        priority = 8
    elif "notice" in url_lower or "notification" in url_lower or "circular" in url_lower:
        doc_type = "notice"
        priority = 7
    elif "facult" in url_lower or "department" in url_lower:
        doc_type = "faculty"
        priority = 6
    elif "program" in url_lower or "course" in url_lower:
        doc_type = "program"
        priority = 6
    elif "news" in url_lower or "event" in text_lower:
        doc_type = "event"
        priority = 3
    else:
        doc_type = "general"
        priority = 1

    return {
        "url": url,
        "section": section,
        "type": doc_type,
        "year": latest_year,
        "priority": priority,
    }

# =========================
# CORE EMBED FUNCTION
# Used by both standalone mode and server.py
# =========================
def embed_pages(
    pages: list[dict],
    collection,
    model: Optional[SentenceTransformer] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    """
    Embed a list of page dicts into ChromaDB collection.

    Args:
        pages: list of {url, content, section} dicts
        collection: ChromaDB collection instance
        model: SentenceTransformer model (loaded if not provided)
        progress_callback: optional function for live progress logging

    Returns:
        Total number of chunks stored
    """

    def log(msg: str):
        print(msg, flush=True)
        if progress_callback:
            progress_callback(msg)

    if model is None:
        log("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    batch_docs = []
    batch_embs = []
    batch_metas = []
    batch_ids = []

    total = 0
    skipped_duplicates = 0
    seen_hashes: set[str] = set()

    for i, page in enumerate(pages):
        url = page.get("url", "unknown")
        raw = page.get("content", "")
        section = page.get("section", None)

        cleaned = clean_text(raw)

        if len(cleaned.split()) < 30:
            continue

        chunks = chunk_text(cleaned)

        if not chunks:
            continue

        meta = extract_metadata(url, cleaned, section)

        try:
            embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        except Exception as e:
            log(f"  [EMBED ERROR] {url}: {e}")
            continue

        log(f"[{i+1}/{len(pages)}] {url} → {len(chunks)} chunks")

        for chunk, emb in zip(chunks, embeddings):
            h = hashlib.md5(chunk.encode()).hexdigest()

            if h in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(h)

            batch_docs.append(chunk)
            batch_embs.append(emb)
            batch_metas.append(meta)
            batch_ids.append(h)
            total += 1

            if len(batch_docs) >= BATCH_SIZE:
                collection.upsert(
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                log(f"Stored {total} chunks...")
                batch_docs = []
                batch_embs = []
                batch_metas = []
                batch_ids = []

    # Final flush
    if batch_docs:
        collection.upsert(
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
            ids=batch_ids
        )

    log(f"Embedding complete. Total: {total} chunks stored, {skipped_duplicates} duplicates skipped.")
    return total

# =========================
# STANDALONE ENTRY POINT
# Run directly: python embed.py
# =========================
if __name__ == "__main__":
    print("Loading model...", flush=True)
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_or_create_collection(name="lumhs")

    print("Loading scraped data...", flush=True)
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        _data = json.load(f)

    print(f"Total pages loaded: {len(_data)}", flush=True)

    _total = embed_pages(_data, _collection, model=_model)

    print(f"\nDONE — {_total} chunks stored.")