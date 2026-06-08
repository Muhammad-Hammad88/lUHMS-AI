import json
import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib

print("Loading model...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="lumhs")

print("Loading scraped data...", flush=True)

with open('scraped_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total pages loaded: {len(data)}", flush=True)

# =========================
# NOISE FILTER
# =========================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_noise(text):
    noise_keywords = [
        "gateway to a brighter future",
        "copyright",
        "all rights reserved",
        "menu",
        "home about contact",
        "student portal",
        "staff portal"
    ]
    t = text.lower()
    return any(k in t for k in noise_keywords)

# =========================
# IMPROVED CHUNKING
# Splits by natural boundaries (paragraphs, sentences)
# instead of blind word count cuts
# =========================
def chunk_text(text, max_words=180, overlap=20):
    # First try to split by paragraph breaks
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\.\s{2,}', text) if p.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = para.split()
        para_len = len(para_words)

        # If single paragraph is too long, split it by sentences
        if para_len > max_words:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_words = sentence.split()
                s_len = len(s_words)

                if current_words + s_len > max_words and current_chunk:
                    chunk = " ".join(current_chunk)
                    if len(chunk.split()) > 20 and not is_noise(chunk):
                        chunks.append(chunk)
                    # overlap — keep last few words for context continuity
                    overlap_words = current_chunk[-overlap:] if overlap else []
                    current_chunk = overlap_words + s_words
                    current_words = len(current_chunk)
                else:
                    current_chunk.extend(s_words)
                    current_words += s_len
        else:
            if current_words + para_len > max_words and current_chunk:
                chunk = " ".join(current_chunk)
                if len(chunk.split()) > 20 and not is_noise(chunk):
                    chunks.append(chunk)
                overlap_words = current_chunk[-overlap:] if overlap else []
                current_chunk = overlap_words + para_words
                current_words = len(current_chunk)
            else:
                current_chunk.extend(para_words)
                current_words += para_len

    # flush remaining
    if current_chunk:
        chunk = " ".join(current_chunk)
        if len(chunk.split()) > 20 and not is_noise(chunk):
            chunks.append(chunk)

    # fallback — if no paragraph splits worked, use word count
    if not chunks:
        words = text.split()
        for i in range(0, len(words), max_words - overlap):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) > 20 and not is_noise(chunk):
                chunks.append(chunk)

    return chunks

# =========================
# METADATA EXTRACTION
# =========================
def extract_metadata(url, text):
    url_lower = url.lower()
    text_lower = text.lower()

    meta = {
        "url": url,
        "year": 0,
        "type": "general",
        "priority": 1
    }

    # detect year — pick most recent
    years = re.findall(r"(20\d{2})", text_lower)
    if years:
        meta["year"] = max([int(y) for y in years])

    # detect type from URL first then text
    if "admission" in url_lower or "admission" in text_lower:
        meta["type"] = "admission"
        meta["priority"] = 10
    elif "result" in url_lower:
        meta["type"] = "result"
        meta["priority"] = 8
    elif "notice" in url_lower or "notification" in url_lower or "circular" in url_lower:
        meta["type"] = "notice"
        meta["priority"] = 7
    elif "faculty" in url_lower or "department" in url_lower:
        meta["type"] = "faculty"
        meta["priority"] = 6
    elif "program" in url_lower or "course" in url_lower:
        meta["type"] = "program"
        meta["priority"] = 6
    elif "fee" in url_lower or "fee" in text_lower:
        meta["type"] = "fee"
        meta["priority"] = 8
    elif "event" in text_lower or "news" in url_lower:
        meta["type"] = "event"
        meta["priority"] = 3
    elif url_lower.endswith(".pdf"):
        meta["type"] = "document"
        meta["priority"] = 9

    return meta

# =========================
# STORAGE
# =========================
batch_docs = []
batch_embs = []
batch_metas = []
batch_ids = []

BATCH_SIZE = 64
total = 0
seen_hashes = set()

print("Embedding data...", flush=True)

for i, page in enumerate(data):
    url = page.get("url", "unknown")
    raw = page.get("content", "")

    cleaned = clean_text(raw)

    if len(cleaned.split()) < 30:
        continue

    chunks = chunk_text(cleaned)

    if not chunks:
        continue

    meta = extract_metadata(url, cleaned)

    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    print(f"[{i+1}/{len(data)}] {url} -> {len(chunks)} chunks", flush=True)

    for chunk, emb in zip(chunks, embeddings):
        h = hashlib.md5(chunk.encode()).hexdigest()

        if h in seen_hashes:
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
            print(f"Stored {total} chunks...", flush=True)
            batch_docs = []
            batch_embs = []
            batch_metas = []
            batch_ids = []

# final flush
if batch_docs:
    collection.upsert(
        documents=batch_docs,
        embeddings=batch_embs,
        metadatas=batch_metas,
        ids=batch_ids
    )

print("\nDONE")
print("Total chunks stored:", total)