"""
MONOKO — RAG API Server
=======================
Wraps the MonokoRAG FAISS engine in a FastAPI server so the production
index.html can call it from the browser.

Endpoints:
    GET  /health           — liveness check
    POST /api/context      — returns FAISS-retrieved context string for a query

Usage:
    pip install fastapi uvicorn[standard] langdetect
    uvicorn rag_api:app --host 0.0.0.0 --port 8000

    # Or for development with auto-reload:
    uvicorn rag_api:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Make sure step3_build_rag imports work from this directory ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_build_rag import MonokoRAG

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="Monoko RAG API", version="1.0.0")

# Allow requests from any origin (index.html may be served from file:// or
# any domain — Supabase CDN, GitHub Pages, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load RAG once at startup ──────────────────────────────────────────────────
rag: MonokoRAG | None = None

@app.on_event("startup")
def load_rag():
    global rag
    print("\n🚀 Starting Monoko RAG API...")
    try:
        rag = MonokoRAG()
        print("✅ RAG ready\n")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Run:  python step3_build_rag.py   to build the indexes first.\n")
        # Don't crash — health endpoint should still respond so the deploy
        # platform can show the error without restart-looping.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContextRequest(BaseModel):
    query: str
    top_k: int = 20          # how many results to retrieve from FAISS


class ContextResponse(BaseModel):
    context: str             # formatted string injected into LLM system prompt
    query_lang: str          # "fr" or "ln" — which index was used
    result_count: int        # how many docs were retrieved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/health")
def health():
    return {
        "status": "ok" if rag is not None else "index_not_loaded",
        "vectors": rag.index_fr.ntotal if rag else 0,
    }


@app.post("/api/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index not loaded. Run step3_build_rag.py first.",
        )

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    # 1. Detect language and retrieve from the right FAISS index
    query_lang = rag._detect_language(query)
    results = rag.retrieve(query, top_k=req.top_k)

    # 2. Build context string (same format as the production app expects)
    words     = [r for r in results if r.get("type") == "word"]
    sentences = [r for r in results if r.get("type") != "word"]

    parts = []

    if words:
        parts.append("=== VOCABULAIRE VÉRIFIÉ ===")
        for w in words[:10]:
            quality_tag = " [vérifié par professeur]" if w.get("quality") == "verified" else ""
            parts.append(f"• {w['french']} → {w['lingala']}{quality_tag}")

    if sentences:
        parts.append("\n=== PHRASES PARALLÈLES ===")
        for s in sentences[:20]:
            if s.get("quality") == "verified":
                quality_tag = " [vérifié par professeur]"
            elif s.get("quality") == "gold":
                quality_tag = " [gold]"
            else:
                quality_tag = ""
            parts.append(f"FR: {s['french']}")
            parts.append(f"LN: {s['lingala']}{quality_tag}")
            parts.append("")

    context = "\n".join(parts) if parts else ""

    return ContextResponse(
        context=context,
        query_lang=query_lang,
        result_count=len(results),
    )
