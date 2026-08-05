"""
LUMHS Chatbot API Server (Knowledge Base Version)
- Replaces ChromaDB + static_answers.json with a single knowledge_base.json
- SIMPLE/COMPLEX classification via fast keyword matching (no LLM call)
- Section detection via keyword/tag matching + embedding similarity fallback
- Admin endpoints for knowledge base management
- Analytics tracking for conversations, programs, and unanswered queries
"""

import asyncio
import itertools
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

import numpy as np
from numpy.linalg import norm

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import httpx
from threading import Lock
import uvicorn

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
# ADMIN CONFIG
# =========================
ADMIN_USERNAME = "123"
ADMIN_PASSWORD = "123"
SESSION_EXPIRY_MINUTES = 30
SUGGESTIONS_FILE = 'suggestions.json'

# =========================
# OLLAMA CONFIG
# =========================
OLLAMA_URLS = [
    "http://localhost:11434/api/generate",
]
OLLAMA_MODEL = "llama3.2:3b"
ollama_cycle = itertools.cycle(OLLAMA_URLS)

# =========================
# KNOWLEDGE BASE CONFIG
# =========================
KNOWLEDGE_BASE_FILE = 'knowledge_base.json'
CHAT_HISTORY_FILE = 'chat_history.json'
CLASSIFICATION_TIMEOUT = 15
ANSWER_TIMEOUT = 120

# =========================
# CONTEXT CAP CONFIG (Fix #2 - prevents dumping huge JSON into prompt)
# =========================
MAX_CONTEXT_SECTIONS = 3              # max number of sections/programs sent to LLM
MAX_CHARS_PER_SECTION_SIMPLE = 1600    # cap for simple factual questions
MAX_CHARS_PER_SECTION_COMPLEX = 2200   # cap for comparisons/recommendations - allow more detail

# =========================
# CLASSIFICATION KEYWORDS (Fix #3 - skip LLM call for classification)
# =========================
COMPLEX_KEYWORDS = [
    "compare", "comparison", "vs", "versus", "which is better", "which one",
    "which program", "which course", "should i choose", "should i study",
    "should i pick", "should i do", "should i go for", "recommend",
    "recommendation", "suggest", "suggestion", "suits me", "suit me",
    "best for me", "best option", "what should i", "help me decide",
    "help me choose", "difference between", "differences", "pros and cons",
    "based on my", "for someone like me", "i am interested in",
    "i'm interested in", "i like biology", "i like chemistry", "career advice",
    "which suits", "your opinion", "what do you think", "better option",
    "or should i"
]

# =========================
# GENERAL CONFIG
# =========================
MAX_HISTORY = 4

# =========================
# GLOBAL STATE
# =========================
embed_model: Optional[SentenceTransformer] = None
knowledge_base: Dict = {}

# Pre-computed section embeddings for fallback matching
SECTION_EMBEDDINGS: Dict[str, np.ndarray] = {}
SECTION_KEYS: List[str] = []

state_lock = Lock()
session_store: Dict[str, List[Dict]] = {}
admin_sessions: Dict[str, datetime] = {}

GPU_SEMAPHORE = asyncio.Semaphore(10)

# =========================
# ANALYTICS
# =========================
analytics_total_convos = 0
analytics_unanswered: List[str] = []      # stores last 100 unanswered queries
analytics_program_counts: Dict[str, int] = {}  # program_key -> hit count

# =========================
# KNOWLEDGE BASE LOADING
# =========================
def load_knowledge_base() -> Dict:
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        logger.warning(f"{KNOWLEDGE_BASE_FILE} not found!")
        return {}
    try:
        with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded knowledge base with sections: {list(data.get('sections', {}).keys())}")
        return data
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return {}

def save_knowledge_base(data: Dict):
    with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def preload_section_embeddings():
    """Pre-compute embeddings for all section tags/keywords/phrases."""
    global SECTION_EMBEDDINGS, SECTION_KEYS
    if embed_model is None or not knowledge_base:
        return

    sections = knowledge_base.get("sections", {})
    SECTION_KEYS = []
    embeddings_list = []

    for section_key, section_data in sections.items():
        if section_key == "programs":
            for prog_key, prog_data in section_data.items():
                if isinstance(prog_data, dict):
                    combined = " ".join([
                        prog_key,
                        " ".join(prog_data.get("tags", [])),
                        " ".join(prog_data.get("keywords", [])),
                        " ".join(prog_data.get("phrases", [])),
                        prog_data.get("name", ""),
                        prog_data.get("description", "") or ""
                    ])
                    emb = embed_model.encode(combined, normalize_embeddings=True)
                    SECTION_KEYS.append(f"programs.{prog_key}")
                    embeddings_list.append(emb)
        else:
            if isinstance(section_data, dict):
                combined = " ".join([
                    section_key,
                    " ".join(section_data.get("tags", [])),
                    " ".join(section_data.get("keywords", [])),
                    " ".join(section_data.get("phrases", [])),
                ])
                emb = embed_model.encode(combined, normalize_embeddings=True)
                SECTION_KEYS.append(section_key)
                embeddings_list.append(emb)

    for key, emb in zip(SECTION_KEYS, embeddings_list):
        SECTION_EMBEDDINGS[key] = emb

    logger.info(f"Preloaded embeddings for {len(SECTION_KEYS)} sections/programs")

# =========================
# STARTUP / SHUTDOWN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, knowledge_base
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loading knowledge base...")
    knowledge_base = load_knowledge_base()
    preload_section_embeddings()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down server...")

# =========================
# APP
# =========================
app = FastAPI(title="LUMHS Chatbot API", lifespan=lifespan)

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

class SuggestionAddRequest(BaseModel):
    label: str
    question: str
    answer: str
    paraphrase: bool
    token: str

class SuggestionDeleteRequest(BaseModel):
    id: str
    token: str

class KnowledgeUpdateRequest(BaseModel):
    token: str
    section: str
    key: str
    value: str

class ProgramAddRequest(BaseModel):
    token: str
    program_key: str
    program_data: Dict

class ProgramDeleteRequest(BaseModel):
    token: str
    program_key: str

# =========================
# ADMIN AUTH
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
# SUGGESTIONS
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
# CHAT HISTORY (persistent, for admin panel)
# =========================
def load_chat_history() -> List[Dict]:
    """Reads the JSONL chat history file. Each line is one Q&A record."""
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    records = []
    try:
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
    return records

def log_chat_record(session_id: str, question: str, answer: str,
                     question_type: str = "N/A", source: str = "llm"):
    """
    Append one Q&A record to the persistent chat history file (JSONL - one
    JSON object per line). This is append-only, so it stays fast even with
    thousands of records, unlike rewriting the whole file every message.
    source: 'llm' | 'suggestion_chip' | 'fallback_no_match'
    """
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "type": question_type,
        "source": source
    }
    try:
        with state_lock:
            with open(CHAT_HISTORY_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to log chat record: {e}")

# =========================
# SECTION DETECTION
# =========================
def detect_sections(question: str) -> List[str]:
    """
    Detect relevant sections from knowledge_base using:
    1. Keyword/tag matching (instant)
    2. Embedding similarity fallback
    Returns list of matched section keys (e.g. ['programs.mbbs', 'admissions'])
    """
    q_lower = question.lower()
    sections = knowledge_base.get("sections", {})

    # Check for top-level programs tags/keywords/phrases
    if "programs" in sections:
        prog_section = sections["programs"]
        prog_tags = prog_section.get("tags", [])
        prog_keywords = prog_section.get("keywords", [])
        prog_phrases = prog_section.get("phrases", [])
        all_terms = prog_tags + prog_keywords

        if any(phrase.lower() in q_lower for phrase in prog_phrases):
            matched_programs = [
                f"programs.{k}" for k in prog_section.keys()
                if k not in ("tags", "keywords", "phrases")
            ]
            logger.info(f"General programs phrase match → all programs: {matched_programs}")
            return matched_programs

        if any(term.lower() in q_lower for term in all_terms if term):
            matched_programs = [
                f"programs.{k}" for k in prog_section.keys()
                if k not in ("tags", "keywords", "phrases")
            ]
            logger.info(f"General programs tag/keyword match → all programs: {matched_programs}")
            return matched_programs

    # Per-section matching
    matched = []
    for section_key, section_data in sections.items():
        if section_key == "programs":
            for prog_key, prog_data in section_data.items():
                if prog_key in ("tags", "keywords", "phrases"):
                    continue
                if not isinstance(prog_data, dict):
                    continue
                all_terms = (
                    prog_data.get("tags", []) +
                    prog_data.get("keywords", []) +
                    [prog_key, prog_data.get("name", "")]
                )
                phrases = prog_data.get("phrases", [])
                if any(phrase.lower() in q_lower for phrase in phrases):
                    matched.append(f"programs.{prog_key}")
                    continue
                if any(term.lower() in q_lower for term in all_terms if term):
                    matched.append(f"programs.{prog_key}")
        else:
            if not isinstance(section_data, dict):
                continue
            all_terms = (
                section_data.get("tags", []) +
                section_data.get("keywords", []) +
                [section_key]
            )
            phrases = section_data.get("phrases", [])
            if any(phrase.lower() in q_lower for phrase in phrases):
                matched.append(section_key)
                continue
            if any(term.lower() in q_lower for term in all_terms if term):
                matched.append(section_key)

    if matched:
        logger.info(f"Keyword match found sections: {matched}")
        return matched

    # Embedding similarity fallback
    if embed_model is None or not SECTION_KEYS:
        return []

    q_emb = embed_model.encode(question, normalize_embeddings=True)
    scores = {}
    for key in SECTION_KEYS:
        emb = SECTION_EMBEDDINGS.get(key)
        if emb is not None:
            score = float(np.dot(q_emb, emb))
            scores[key] = score

    if not scores:
        return []

    sorted_sections = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [k for k, s in sorted_sections[:2] if s > 0.3]
    logger.info(f"Embedding fallback found sections: {top}")
    return top

def get_section_context(section_keys: List[str], is_complex: bool = False) -> str:
    """
    Build context string from matched section keys.

    FIX #2: Capped so we never dump huge amounts of JSON into the prompt.
    - Only the first MAX_CONTEXT_SECTIONS matched sections are used.
    - Each section's content is truncated based on complexity:
      SIMPLE questions get a tight cap (fast), COMPLEX questions (comparisons,
      recommendations) get a larger cap since they genuinely need more detail
      per program to give a useful answer.
    """
    sections = knowledge_base.get("sections", {})
    context_parts = []

    char_cap = MAX_CHARS_PER_SECTION_COMPLEX if is_complex else MAX_CHARS_PER_SECTION_SIMPLE

    # Cap number of sections sent to the LLM
    capped_keys = section_keys[:MAX_CONTEXT_SECTIONS]
    if len(section_keys) > MAX_CONTEXT_SECTIONS:
        logger.info(
            f"Context capped: {len(section_keys)} sections matched, "
            f"only using first {MAX_CONTEXT_SECTIONS}: {capped_keys}"
        )

    for key in capped_keys:
        if key.startswith("programs."):
            prog_key = key.split(".", 1)[1]
            prog_data = sections.get("programs", {}).get(prog_key, {})
            if prog_data:
                content = {k: v for k, v in prog_data.items()
                          if k not in ("tags", "keywords", "phrases")}
                text = json.dumps(content, indent=2)
                if len(text) > char_cap:
                    text = text[:char_cap]
                context_parts.append(f"[{prog_key.upper()} Program Information]\n{text}")
        else:
            section_data = sections.get(key, {})
            if section_data:
                content = {k: v for k, v in section_data.items()
                          if k not in ("tags", "keywords", "phrases")}
                text = json.dumps(content, indent=2)
                if len(text) > char_cap:
                    text = text[:char_cap]
                context_parts.append(f"[{key.upper()} Information]\n{text}")

    final_context = "\n\n".join(context_parts)
    logger.info(f"CONTEXT chars: {len(final_context)} (~{len(final_context)//4} tokens est., cap={char_cap})")
    return final_context

# =========================
# CLASSIFICATION (Fix #3 - no LLM call, instant keyword matching)
# =========================
async def classify_question(question: str) -> str:
    """
    Returns 'SIMPLE' or 'COMPLEX'.

    FIX #3: Previously this made a separate Ollama call for every single
    question, costing ~2-3s of wasted time even for simple factual questions.
    Since the knowledge base is already tagged/structured, we can reliably
    detect COMPLEX questions (comparisons, recommendations, personalized
    advice) using keyword matching instead - instant, no LLM round-trip.
    """
    q_lower = question.lower()

    if any(kw in q_lower for kw in COMPLEX_KEYWORDS):
        logger.info("Classification (keyword): COMPLEX")
        return "COMPLEX"

    logger.info("Classification (keyword): SIMPLE")
    return "SIMPLE"

# =========================
# HISTORY
# =========================
def build_history_text(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-MAX_HISTORY:]:
        lines.append(f"User: {h['question']}")
        lines.append(f"Assistant: {h['answer']}")
    return "\n".join(lines)

# =========================
# LLM CALL
# =========================
async def call_llm(question: str, context: str, history: List[Dict], is_complex: bool = False) -> str:
    history_text = build_history_text(history)
    history_section = f"\nCONVERSATION HISTORY:\n{history_text}\n" if history_text else ""

    if is_complex:
        task_instruction = """
Answer using ONLY the information provided in the CONTEXT.

- Answer immediately.
- Be detailed, organized, and helpful.
- If comparing programs, clearly highlight the key differences.
- If recommending, only consider the user's background or interests mentioned in the conversation.
- Use headings, bullet points, or numbered lists when they improve readability.
- Preserve facts exactly as written.
- Never infer, assume, estimate, or fabricate information.
"""
    else:
        task_instruction = """
Answer using ONLY the information provided in the CONTEXT.

- Answer immediately.
- Be concise, direct, and natural.
- Use numbered lists only when helpful.
- Preserve facts exactly as written.
- Never infer, assume, estimate, or fabricate information.
"""

    prompt = f"""You are LUMHS Assistant, the AI assistant for Liaquat University of Medical and Health Sciences (LUMHS), Jamshoro, Pakistan.

IDENTITY:
- Introduce yourself ONLY if the user asks who you are or what you are.
- Never greet unless the user greets first.
- Never start any answer with a greeting or self-introduction.
- Start directly with the answer.

{history_section}

STRICT RULES:

- Answer ONLY using facts explicitly present in the CONTEXT.
- Never invent, assume, estimate, infer, or guess information.
- Never combine separate facts to create new information.

- NEVER mention:
  - context
  - provided context
  - provided data
  - provided information
  - documents
  - document
  - retrieval
  - database
  - embeddings
  - sources
  - knowledge base

- NEVER start an answer with phrases such as:
  - Here's the information...
  - Based on the context...
  - Based on the provided context...
  - Based on the provided data...
  - According to the context...
  - According to the provided information...
  - From the provided data...
  - From the context...
  - The context states...
  - The document states...
  - The provided information states...
  - I found...
  - Below is...
  - Certainly!
  - Sure!
  - Of course!

- NEVER apologize.
- NEVER ask the user for:
  - more context
  - more information
  - clarification
  - additional details
  - another document
  - another file

- NEVER end an answer with:
  - Hope this helps.
  - Let me know if you have any questions.
  - Let me know if you need anything else.
  - Feel free to ask.
  - Is there anything else I can help with?

- If the requested information is NOT present in the CONTEXT, reply EXACTLY:
I don't have that information right now. You can contact LUMHS at +92 22 9213305 or email registrar@lumhs.edu.pk

- Use conversation history ONLY to resolve follow-up questions.
- Answer ONLY about LUMHS unless the user explicitly asks about another institution.
- If the answer is a list, table, steps, or fees, output ONLY the requested information with no introduction or conclusion.
- Begin every response immediately with the answer.

TASK:
{task_instruction}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=ANSWER_TIMEOUT) as client:
                r = await client.post(
                    next(ollama_cycle),
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_thread": 8, "num_batch": 512, "num_ctx": 4096}
                    }
                )
            elapsed = time.time() - start_time
            logger.info(f"LLM TIME: {elapsed:.1f}s")
            data = r.json()
            response_text = data.get("response", "").strip()
            if not response_text:
                raise RuntimeError("Ollama returned empty response")
            return response_text
        except Exception as e:
            logger.error(f"Ollama attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2)
            else:
                return "I couldn't generate a response right now. Please try again or contact LUMHS at +92 22 9213305"

    return "I couldn't generate a response right now. Please try again or contact LUMHS at +92 22 9213305"

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/ask")
async def ask(query: ChatQuery):
    global analytics_total_convos

    q = query.question.strip()
    q_lower = q.lower()
    session_id = query.session_id or str(uuid.uuid4())

    # Track total conversations
    with state_lock:
        analytics_total_convos += 1

    # 1. Check suggestion chips (exact match)
    suggestions = load_suggestions()
    matched_faq = None
    for s in suggestions:
        if q_lower == s["question"].lower() or q_lower == s["label"].lower():
            matched_faq = s
            break

    if matched_faq and matched_faq["answer"].strip().upper() not in ("AUTO", "AUTO_GENERATE"):
        if not matched_faq.get("paraphrase", False):
            with state_lock:
                if session_id not in session_store:
                    session_store[session_id] = []
                session_store[session_id].append({"question": q, "answer": matched_faq["answer"]})
                session_store[session_id] = session_store[session_id][-MAX_HISTORY:]
            log_chat_record(session_id, q, matched_faq["answer"], question_type="N/A", source="suggestion_chip")
            return {
                "answer": matched_faq["answer"],
                "sources": ["(Official Predefined Answer)"],
                "status": "success",
                "session_id": session_id
            }

    # 2. Detect relevant sections first (needed for smarter classification below)
    matched_sections = detect_sections(q)

    # 3. Classify question (fast keyword-based, no LLM call)
    # Safety net: if 2+ programs are matched at once (e.g. "bds vs mbbs",
    # "bds mbbs difference"), it's a comparison regardless of exact wording -
    # no need to rely on catching every possible keyword.
    matched_program_count = sum(1 for k in matched_sections if k.startswith("programs."))
    if matched_program_count >= 2:
        is_complex = True
        question_type = "COMPLEX"
        logger.info(f"Classification (multi-program match): COMPLEX ({matched_program_count} programs)")
    else:
        question_type = await classify_question(q)
        is_complex = question_type == "COMPLEX"

    if not matched_sections:
        # Track unanswered query
        with state_lock:
            analytics_unanswered.append(q)
            analytics_unanswered[:] = analytics_unanswered[-100:]

        fallback = "I don't have specific information about that. Please contact LUMHS directly at +92-22-9213305 or visit www.lumhs.edu.pk"
        with state_lock:
            if session_id not in session_store:
                session_store[session_id] = []
            session_store[session_id].append({"question": q, "answer": fallback})
            session_store[session_id] = session_store[session_id][-MAX_HISTORY:]
        log_chat_record(session_id, q, fallback, question_type="N/A", source="fallback_no_match")
        return {
            "answer": fallback,
            "sources": [],
            "status": "success",
            "session_id": session_id
        }

    # 4. Track program hits (only if the program is actually mentioned)
    with state_lock:
        for key in matched_sections:
            if not key.startswith("programs."):
                continue

            prog = key.split(".", 1)[1]
            prog_data = knowledge_base["sections"]["programs"].get(prog, {})

            terms = [prog, prog_data.get("name", "")] + \
                    prog_data.get("tags", []) + \
                    prog_data.get("keywords", [])

            if any(term and term.lower() in q_lower for term in terms):
                analytics_program_counts[prog] = analytics_program_counts.get(prog, 0) + 1

    # 5. Build context from matched sections (capped - see Fix #2)
    context = get_section_context(matched_sections, is_complex=is_complex)

    # 6. Get history
    with state_lock:
        history = session_store.get(session_id, [])

    # 7. Call LLM
    async with GPU_SEMAPHORE:
        answer = await call_llm(q, context, history, is_complex=is_complex)

    # 8. Save to session
    with state_lock:
        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append({"question": q, "answer": answer})
        session_store[session_id] = session_store[session_id][-MAX_HISTORY:]

    log_chat_record(session_id, q, answer, question_type=question_type, source="llm")

    return {
        "answer": answer,
        "sources": ["(Official LUMHS Knowledge Base)"],
        "status": "success",
        "session_id": session_id
    }

# =========================
# FRONTEND SERVING
# =========================
@app.get("/")
async def serve_root():
    return FileResponse("widget.html")

@app.get("/widget.html")
async def serve_widget():
    return FileResponse("widget.html")

@app.get("/admin.html")
async def serve_admin():
    return FileResponse("admin.html")

@app.get("/bot-logo.png")
async def serve_logo():
    return FileResponse("bot-logo.png")

# =========================
# ADMIN AUTH ENDPOINTS
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

# =========================
# SUGGESTIONS ENDPOINTS
# =========================
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

# =========================
# KNOWLEDGE BASE ENDPOINTS
# =========================
@app.get("/admin/knowledge")
async def get_knowledge(token: str):
    require_admin(token)
    return knowledge_base

@app.put("/admin/knowledge/section")
async def update_knowledge_field(req: KnowledgeUpdateRequest):
    require_admin(req.token)
    global knowledge_base
    sections = knowledge_base.get("sections", {})
    if req.section not in sections:
        raise HTTPException(status_code=404, detail=f"Section '{req.section}' not found")
    sections[req.section][req.key] = req.value
    knowledge_base["sections"] = sections
    save_knowledge_base(knowledge_base)
    preload_section_embeddings()
    return {"success": True}

@app.post("/admin/knowledge/programs")
async def add_program(req: ProgramAddRequest):
    require_admin(req.token)
    global knowledge_base
    sections = knowledge_base.get("sections", {})
    if "programs" not in sections:
        sections["programs"] = {}
    sections["programs"][req.program_key] = req.program_data
    knowledge_base["sections"] = sections
    save_knowledge_base(knowledge_base)
    preload_section_embeddings()
    return {"success": True}

@app.delete("/admin/knowledge/programs/{program_key}")
async def delete_program(program_key: str, token: str):
    require_admin(token)
    global knowledge_base
    sections = knowledge_base.get("sections", {})
    if program_key not in sections.get("programs", {}):
        raise HTTPException(status_code=404, detail=f"Program '{program_key}' not found")
    del sections["programs"][program_key]
    knowledge_base["sections"] = sections
    save_knowledge_base(knowledge_base)
    preload_section_embeddings()
    return {"success": True}

# =========================
# CHAT HISTORY ENDPOINT (for admin panel)
# =========================
@app.get("/admin/chat-history")
async def api_get_chat_history(token: str, limit: int = 50, offset: int = 0, search: str = ""):
    """
    Returns chat history, newest first.
    - limit/offset: pagination
    - search: optional case-insensitive filter on question or answer text
    """
    require_admin(token)
    history = load_chat_history()
    history.reverse()  # newest first

    if search:
        search_lower = search.lower()
        history = [
            h for h in history
            if search_lower in h.get("question", "").lower()
            or search_lower in h.get("answer", "").lower()
        ]

    total = len(history)
    page = history[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": page
    }

@app.delete("/admin/chat-history")
async def api_clear_chat_history(token: str):
    """Clears all chat history. Use with care - this cannot be undone."""
    require_admin(token)
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")

# =========================
# ANALYTICS ENDPOINT
# =========================
@app.get("/admin/analytics")
async def api_get_analytics(token: str):
    require_admin(token)
    with state_lock:
        top_prog = max(analytics_program_counts, key=lambda k: analytics_program_counts[k]) if analytics_program_counts else "None"
        return {
            "total_conversations": analytics_total_convos,
            "top_program": top_prog,
            "program_counts": dict(sorted(analytics_program_counts.items(), key=lambda x: x[1], reverse=True)),
            "unanswered_count": len(analytics_unanswered),
            "unanswered_queries": analytics_unanswered
        }

# =========================
# HEALTH
# =========================
@app.get("/health")
async def health():
    sections = knowledge_base.get("sections", {})
    programs = sections.get("programs", {})
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "knowledge_base_sections": list(sections.keys()),
        "programs_count": len(programs),
        "total_conversations": analytics_total_convos,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)