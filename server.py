"""
LUMHS Chatbot API Server (Async Thread-Pool Version)
"""

import asyncio
import logging
import re
import uuid
import time
import hashlib
import secrets
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from threading import Lock
import uvicorn
import google.generativeai as genai

# =========================
# LOGGING
# =========================
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LUMHS_SERVER")

# =========================
# ADMIN CONFIG & FILES
# =========================
ADMIN_USERNAME = "123"
ADMIN_PASSWORD = "123"
SESSION_EXPIRY_MINUTES = 30
SUGGESTIONS_FILE = 'suggestions.json'

# =========================
# GEMINI CONFIG
# =========================
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-3.5-flash"

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# =========================
# RAG CONFIG
# =========================
MAX_HISTORY = 4
MAX_CONTEXT_WORDS = 800
MAX_RETRIEVAL_RESULTS = 15

# =========================
# GLOBAL STATE
# =========================
embed_model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.PersistentClient] = None
collection = None

state_lock = Lock()
query_cache: Dict[str, Any] = {}
session_store: Dict[str, List[Dict]] = {}
admin_sessions: Dict[str, datetime] = {}
active_job: Optional[Dict] = None
job_logs: Dict[str, List[str]] = {}

GPU_SEMAPHORE = asyncio.Semaphore(20)
UPDATE_LOCK = asyncio.Lock()

# =========================
# STARTUP / SHUTDOWN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, chroma_client, collection
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("lumhs")
    logger.info("Server ready.")
    yield
    logger.info("Shutting down server...")

# =========================
# APP
# =========================
app = FastAPI(title="LUMHS RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SCHEMAS
# =========================
class ChatQuery(BaseModel):
    question: str
    session_id: str = ""

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class UpdateRequest(BaseModel):
    sections: List[str]
    token: str

class FullUpdateRequest(BaseModel):
    token: str

class SuggestionAddRequest(BaseModel):
    label: str
    question: str
    answer: str
    paraphrase: bool
    token: str

class SuggestionDeleteRequest(BaseModel):
    id: str
    token: str

# =========================
# ADMIN AUTH HELPERS
# =========================
def generate_token() -> str:
    return secrets.token_hex(32)

def is_valid_admin_token(token: str) -> bool:
    with state_lock:
        expiry = admin_sessions.get(token)
        if not expiry:
            return False
        if datetime.utcnow() > expiry:
            admin_sessions.pop(token, None)
            return False
        admin_sessions[token] = datetime.utcnow() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
        return True

def require_admin(token: str):
    if not is_valid_admin_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in again.")

# =========================
# SUGGESTION CHIP HELPERS
# =========================
def load_suggestions() -> List[Dict]:
    if not os.path.exists(SUGGESTIONS_FILE):
        return []
    try:
        with open(SUGGESTIONS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_suggestions(data: List[Dict]):
    with open(SUGGESTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# =========================
# EMBEDDING & RETRIEVAL 
# =========================
def embed_text(text: str) -> List[float]:
    return embed_model.encode(text, normalize_embeddings=True).tolist()

def extract_years(text: str) -> List[int]:
    return [int(y) for y in re.findall(r"20\d{2}", text)]

def get_query_year(query: str) -> Optional[int]:
    years = extract_years(query)
    return max(years) if years else None

def rerank_by_recency(docs: List[str], metas: List[Dict]) -> tuple:
    all_years = []
    for doc in docs:
        all_years.extend(extract_years(doc))
    if not all_years:
        return docs, metas
    latest = max(all_years)
    top_two = set(sorted(set(all_years), reverse=True)[:2])
    scored = []
    for doc, meta in zip(docs, metas):
        chunk_years = set(extract_years(doc))
        score = 2 if latest in chunk_years else (1 if chunk_years & top_two else 0)
        scored.append((score, doc, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d, _ in scored], [m for _, _, m in scored]

async def retrieve(query: str) -> tuple[str, List[str]]:
    vector = embed_text(query)
    asked_year = get_query_year(query)

    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[vector],
        n_results=MAX_RETRIEVAL_RESULTS,
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
        if words > MAX_CONTEXT_WORDS:
            break

        context_blocks.append(doc.strip())
        words += len(doc.split())

        url = (meta or {}).get("url", "")
        if url:
            clean_url = str(url).replace("http://", "https://").replace("https://lumhs", "https://www.lumhs").rstrip("/")
            if clean_url not in seen_sources:
                seen_sources.add(clean_url)
                sources.append(clean_url)

    return "\n\n".join(context_blocks), sources

def build_history_text(history: List[Dict]) -> str:
    if not history: return ""
    lines = []
    for h in history[-MAX_HISTORY:]:
        lines.append(f"User: {h['question']}")
        lines.append(f"Assistant: {h['answer']}")
    return "\n".join(lines)

async def call_llm(question: str, context: str, history: List[Dict]) -> str:
    history_text = build_history_text(history)
    history_section = f"\nCONVERSATION HISTORY:\n{history_text}\n" if history_text else ""

    prompt = f"""You are LUMHS Assistant — a smart, friendly AI chatbot for Liaquat University of Medical and Health Sciences (LUMHS), Jamshoro, Pakistan.

IDENTITY:
- Only introduce yourself if someone directly asks "who are you" or "what are you"
- Never start any answer with "LUMHS Assistant here!" or any greeting or self-introduction
- Just answer the question directly and naturally
{history_section}
STRICT RULES:
- Focus answers on LUMHS — if context contains information about other universities or hospitals while answering an LUMHS question, ignore that information
- If someone directly asks about another university say "I'm specifically built for LUMHS and don't have detailed information about other universities"
- ONLY use facts explicitly stated in the CONTEXT below — never invent, assume, or guess anything
- Never say "based on the context", "the text states", "as mentioned" — just answer naturally
- Never calculate or sum up numbers unless user specifically asks for a total
- If something is not in the context say: "I don't have that information right now. You can contact LUMHS at +92 22 9213305 or email registrar@lumhs.edu.pk"
- For admissions, fees, dates — always use the most recent year information available in the context
- Use conversation history to understand follow-up questions and pronouns like "it", "that", "this program"

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

    response = await asyncio.to_thread(gemini.generate_content, prompt)
    return response.text.strip()

# =========================
# BACKGROUND WORKERS
# =========================
def log_progress(job_id: str, message: str):
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    logger.info(entry)
    with state_lock:
        if job_id not in job_logs:
            job_logs[job_id] = []
        job_logs[job_id].append(entry)

def blocking_section_update(job_id: str, sections: List[str]):
    from scraper import LUMHSScraper, load_existing, merge_section_data, save_full
    from embed import embed_pages
    global collection, query_cache, active_job

    try:
        scraper = LUMHSScraper()
        existing_data = load_existing()

        for section in sections:
            log_progress(job_id, f"Initializing scraper for section: {section}...")
            new_pages = scraper.scrape_section(section, progress_callback=lambda msg: log_progress(job_id, msg))
            
            log_progress(job_id, f"Scrape complete. Safely removing old {section} data from Vector DB...")
            try:
                collection.delete(where={"section": section})
                log_progress(job_id, f"Successfully cleared old {section} DB chunks.")
            except Exception as e:
                log_progress(job_id, f"Warning on delete: {e}")

            log_progress(job_id, f"Generating new embeddings for {section}...")
            embed_pages(new_pages, collection, progress_callback=lambda msg: log_progress(job_id, msg))
            
            existing_data = merge_section_data(existing_data, new_pages, section)

        log_progress(job_id, "Saving master JSON records to disk...")
        save_full(existing_data)
        
        with state_lock:
            query_cache.clear()
            
        log_progress(job_id, "___DONE___")
    except Exception as e:
        log_progress(job_id, f"ERROR: {str(e)}")
        log_progress(job_id, "___ERROR___")
    finally:
        with state_lock:
            active_job = None

def blocking_full_update(job_id: str):
    import shutil
    import os
    from scraper import LUMHSScraper, save_full
    from embed import embed_pages
    global chroma_client, collection, query_cache, active_job

    try:
        log_progress(job_id, "Wiping existing ChromaDB local files...")
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        chroma_client = chromadb.PersistentClient(path=chroma_path)
        collection = chroma_client.get_or_create_collection("lumhs")

        scraper = LUMHSScraper()
        log_progress(job_id, "Initiating global site crawl...")
        all_pages = scraper.scrape_full(progress_callback=lambda msg: log_progress(job_id, msg))
        
        log_progress(job_id, "Global scrape complete. Saving master JSON...")
        save_full(all_pages)
        
        log_progress(job_id, "Embedding complete site matrix...")
        embed_pages(all_pages, collection, progress_callback=lambda msg: log_progress(job_id, msg))

        with state_lock:
            query_cache.clear()
            
        log_progress(job_id, "___DONE___")
    except Exception as e:
        log_progress(job_id, f"CRITICAL ERROR: {str(e)}")
        log_progress(job_id, "___ERROR___")
    finally:
        with state_lock:
            active_job = None

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/ask")
async def ask(query: ChatQuery):
    q = query.question.strip()
    q_lower = q.lower()
    session_id = query.session_id or str(uuid.uuid4())

    suggestions = load_suggestions()
    matched_faq = None
    for s in suggestions:
        if q_lower == s["question"].lower() or q_lower == s["label"].lower():
            matched_faq = s
            break

    # If match found, check if it's NOT AUTO_GENERATE
    if matched_faq and matched_faq["answer"].strip().upper() != "AUTO_GENERATE":
        if not matched_faq.get("paraphrase", False):
            with state_lock:
                if session_id not in session_store: session_store[session_id] = []
                session_store[session_id].append({"question": q, "answer": matched_faq["answer"]})
                session_store[session_id] = session_store[session_id][-MAX_HISTORY:]
            return {"answer": matched_faq["answer"], "sources": ["(Official Predefined Answer)"], "status": "success", "session_id": session_id}
        
        # Paraphrase flow
        context = f"OFFICIAL FACTS TO PARAPHRASE: {matched_faq['answer']}"
        sources = ["(Official Predefined Answer)"]
    else:
        # Standard RAG Flow
        context, sources = await retrieve(q)
        if not context:
            return {"answer": "I don't have that information right now.", "sources": [], "status": "success", "session_id": session_id}

    with state_lock:
        history = session_store.get(session_id, [])

    async with GPU_SEMAPHORE:
        answer = await call_llm(q, context, history)

    with state_lock:
        if session_id not in session_store: session_store[session_id] = []
        session_store[session_id].append({"question": q, "answer": answer})
        session_store[session_id] = session_store[session_id][-MAX_HISTORY:]
        # Cache non-history dependent results
        if not history:
            cache_key = hashlib.md5(q_lower.encode()).hexdigest()
            query_cache[cache_key] = {"answer": answer, "sources": sources, "status": "success"}

    return {"answer": answer, "sources": sources, "status": "success", "session_id": session_id}

# =========================
# ADMIN ENDPOINTS
# =========================
@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest):
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = generate_token()
    with state_lock:
        admin_sessions[token] = datetime.utcnow() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
    return {"token": token, "expires_in": SESSION_EXPIRY_MINUTES * 60}

@app.post("/admin/logout")
async def admin_logout(req: dict):
    token = req.get("token", "")
    with state_lock:
        admin_sessions.pop(token, None)
    return {"status": "logged out"}

@app.get("/admin/suggestions")
async def api_get_suggestions():
    return {"suggestions": load_suggestions()}

@app.post("/admin/suggestions/add")
async def api_add_suggestion(req: SuggestionAddRequest):
    require_admin(req.token)
    suggestions = load_suggestions()
    new_item = {
        "id": f"chip_{int(time.time())}",
        "label": req.label.strip(),
        "question": req.question.strip(),
        "answer": req.answer.strip(),
        "paraphrase": req.paraphrase
    }
    suggestions.append(new_item)
    save_suggestions(suggestions)
    return {"success": True, "suggestions": suggestions}

@app.post("/admin/suggestions/delete")
async def api_delete_suggestion(req: SuggestionDeleteRequest):
    require_admin(req.token)
    suggestions = load_suggestions()
    suggestions = [s for s in suggestions if s["id"] != req.id]
    save_suggestions(suggestions)
    return {"success": True, "suggestions": suggestions}

@app.post("/admin/update")
async def admin_update(req: UpdateRequest):
    require_admin(req.token)
    global active_job
    with state_lock:
        if active_job is not None:
            raise HTTPException(status_code=400, detail="Another update is running.")
        
        job_id = str(uuid.uuid4())
        job_logs[job_id] = []
        active_job = {"job_id": job_id, "status": "running"}

    asyncio.create_task(asyncio.to_thread(blocking_section_update, job_id, req.sections))
    return {"status": "started", "job_id": job_id}

@app.post("/admin/update/full")
async def admin_full_update(req: FullUpdateRequest):
    require_admin(req.token)
    global active_job
    with state_lock:
        if active_job is not None:
            raise HTTPException(status_code=400, detail="Another update is running.")
        
        job_id = str(uuid.uuid4())
        job_logs[job_id] = []
        active_job = {"job_id": job_id, "status": "running"}

    asyncio.create_task(asyncio.to_thread(blocking_full_update, job_id))
    return {"status": "started", "job_id": job_id}

@app.get("/admin/progress/{job_id}")
async def admin_progress(job_id: str, token: str):
    require_admin(token)

    async def event_stream():
        sent = 0
        while True:
            with state_lock:
                logs = job_logs.get(job_id, [])
            
            while sent < len(logs):
                line = logs[sent]
                sent += 1
                
                if "___DONE___" in line:
                    yield "data: [DONE]\n\n"
                    return
                elif "___ERROR___" in line:
                    yield "data: [ERROR]\n\n"
                    return
                else:
                    yield f"data: {line}\n\n"
            
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/admin/sections")
async def admin_sections(token: str):
    require_admin(token)
    try:
        from sections import SECTION_LABELS
    except ImportError:
        SECTION_LABELS = {
            "admissions": "Admissions & Faculty",
            "examinations": "Examinations & Results",
            "news": "Latest News & Events"
        }
    return {"sections": [{"key": k, "label": v} for k, v in SECTION_LABELS.items()]}

@app.get("/admin/jobs")
async def get_jobs(token: str):
    require_admin(token)
    with state_lock:
        return {
            "status": "busy" if active_job else "idle", 
            "active_job": active_job
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)