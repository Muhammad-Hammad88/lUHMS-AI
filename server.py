import asyncio
import itertools
import logging
import re
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import httpx
from threading import Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LUMHS_RAG")

app = FastAPI(title="LUMHS RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache_lock = Lock()
query_cache: Dict[str, Any] = {}
session_store: Dict[str, List[Dict]] = {}  # per session memory
GPU_SEMAPHORE = asyncio.Semaphore(20)

MAX_HISTORY = 4  # keep last 4 exchanges per session
SESSION_TIMEOUT = 1800  # 30 minutes, then session cleared

print("Loading models...", flush=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("lumhs")

OLLAMA_URLS = ["http://localhost:11434/api/generate"]
OLLAMA_MODEL = "llama3.2:3b"
ollama_cycle = itertools.cycle(OLLAMA_URLS)

class Query(BaseModel):
    question: str
    session_id: str = ""  # frontend sends this

def embed(text: str):
    return embed_model.encode(text, normalize_embeddings=True).tolist()

def extract_year(text: str):
    years = re.findall(r"20\d{2}", text)
    return [int(y) for y in years]

def extract_year_from_query(q: str):
    years = extract_year(q)
    return max(years) if years else None

def rerank_by_recency(docs, metas):
    all_years = []
    for doc in docs:
        all_years.extend(extract_year(doc))

    if not all_years:
        return docs, metas

    latest_year = max(all_years)
    second_latest = sorted(set(all_years), reverse=True)
    recent_years = set(second_latest[:2])

    scored = []
    for doc, meta in zip(docs, metas):
        years_in_chunk = set(extract_year(doc))
        if latest_year in years_in_chunk:
            score = 2
        elif years_in_chunk & recent_years:
            score = 1
        else:
            score = 0
        scored.append((score, doc, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d, _ in scored], [m for _, _, m in scored]

async def retrieve(query: str):
    vector = embed(query)
    asked_year = extract_year_from_query(query)

    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[vector],
        n_results=15,
        include=["documents", "metadatas", "distances"]
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    if not asked_year:
        docs, metas = rerank_by_recency(docs, metas)

    context_blocks = []
    seen_sources = set()
    sources = []
    words = 0

    for doc, meta in zip(docs, metas):
        if not doc or len(doc.strip()) < 30:
            continue
        if words > 800:
            break

        context_blocks.append(doc.strip())
        words += len(doc.split())

        url = (meta or {}).get("url", "")
        if url is None:
            url = ""
        if not isinstance(url, str):
            try:
                url = str(url)
            except Exception:
                url = ""
        if url:
            clean_url = url.replace("http://", "https://")
            clean_url = clean_url.replace("https://lumhs", "https://www.lumhs")
            clean_url = clean_url.rstrip("/")
            if clean_url not in seen_sources:
                seen_sources.add(clean_url)
                sources.append(clean_url)

    return "\n\n".join(context_blocks), sources

def build_history_text(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-MAX_HISTORY:]:
        lines.append(f"User: {h['question']}")
        lines.append(f"Assistant: {h['answer']}")
    return "\n".join(lines)

async def ask_llm(question: str, context: str, history: list):
    history_text = build_history_text(history)

    history_section = ""
    if history_text:
        history_section = f"""
CONVERSATION HISTORY (for context only):
{history_text}
"""

    prompt = f"""You are LUMHS Assistant — a smart, friendly AI chatbot for Liaquat University of Medical and Health Sciences (LUMHS), Jamshoro, Pakistan.

IDENTITY:
- Only introduce yourself if someone directly asks "who are you" or "what are you"
- Never start any answer with "LUMHS Assistant here!" or any greeting or self-introduction
- Just answer the question directly and naturally

{history_section}

STRICT RULES:
- Answer ONLY about LUMHS Jamshoro unless the user specifically mentions another institution by name
- If context contains information about other universities, hospitals, or institutions (DUHS, JSMU, PNS Shifa, CMH, Navy Hospital, etc.) ignore it unless the user specifically asked about them
- ONLY use facts explicitly stated in the CONTEXT below — never invent, assume, or guess anything
- Never say "based on the context", "the text states", "as mentioned", "according to" — just answer naturally
- Never calculate or sum up numbers unless user specifically asks for a total
- If something is not in the context, say: "I don't have that information right now. You can contact LUMHS at +92 22 9213305 or email registrar@lumhs.edu.pk"
- For admissions, fees, dates — always use the most recent year information available in the context
- Use the conversation history to understand follow-up questions and pronouns like "it", "that", "this program"

FORMATTING:
- Be natural and conversational like a helpful university staff member
- For lists use clean numbered points on separate lines
- Keep answers concise but complete — no unnecessary filler sentences
- Never add "Please note", "I hope this helps", "Feel free to ask" type phrases at the end

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(next(ollama_cycle), json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
    return r.json()["response"].strip()

@app.post("/ask")
async def ask(query: Query):
    q = query.question.strip()
    session_id = query.session_id or str(uuid.uuid4())

    with cache_lock:
        history = session_store.get(session_id, [])

    cache_key = q.lower()
    if not history:
        with cache_lock:
            if cache_key in query_cache:
                logger.info(f"Cache hit: {cache_key}")
                cached = query_cache[cache_key].copy()
                cached["session_id"] = session_id
                return cached

    context, sources = await retrieve(q)

    if not context:
        response = {
            "answer": "I don't have that information right now. You can contact LUMHS at +92 22 9213305 or email registrar@lumhs.edu.pk",
            "sources": [],
            "status": "success",
            "session_id": session_id
        }
        return response

    async with GPU_SEMAPHORE:
        answer = await ask_llm(q, context, history)

    if not answer:
        answer = "I couldn't generate a proper response. Please contact LUMHS at +92 22 9213305"

    response = {
        "answer": answer,
        "sources": sources,
        "status": "success",
        "session_id": session_id
    }

    with cache_lock:
        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append({
            "question": q,
            "answer": answer
        })
        session_store[session_id] = session_store[session_id][-MAX_HISTORY:]

    if not history:
        with cache_lock:
            query_cache[cache_key] = {
                "answer": answer,
                "sources": sources,
                "status": "success"
            }

    return response

    async with GPU_SEMAPHORE:
        answer = await ask_llm(q, context, history)

    if not answer:
        answer = "I couldn't generate a proper response. Please contact LUMHS at +92 22 9213305"

    # Update session history
    with cache_lock:
        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append({
            "question": q,
            "answer": answer
        })
        # Keep only last MAX_HISTORY exchanges
        session_store[session_id] = session_store[session_id][-MAX_HISTORY:]

    # Cache only for fresh single questions
    if not history:
        with cache_lock:
            query_cache[cache_key] = {
                "answer": answer,
                "sources": sources,
                "status": "success"
            }

    return {
        "answer": answer,
        "sources": sources,
        "status": "success",
        "session_id": session_id
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    with cache_lock:
        session_store.pop(session_id, None)
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)