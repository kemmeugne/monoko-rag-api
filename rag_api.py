"""
MONOKO — RAG API Server
=======================
Wraps the MonokoRAG FAISS engine in a FastAPI server so the production
index.html can call it from the browser.

Endpoints:
    GET  /health           — liveness check
    POST /api/context      — returns FAISS-retrieved context string for a query

Usage:
    uvicorn rag_api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_build_rag import MonokoRAG, INDEX_FR_FILE, INDEX_LN_FILE, DOCS_FILE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Supabase Storage URLs for the FAISS index files
# Upload the 3 files to a public Supabase Storage bucket, then set these.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ⚠️  After uploading to Cloudflare R2, replace this with your public bucket URL.
# In R2 dashboard: bucket → Settings → Public Access → copy the r2.dev URL.
# Example: "https://pub-abc123def456.r2.dev"
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://pub-REPLACE_ME.r2.dev")

INDEX_URLS = {
    INDEX_FR_FILE: f"{R2_PUBLIC_URL}/faiss_index_fr.bin",
    INDEX_LN_FILE: f"{R2_PUBLIC_URL}/faiss_index_ln.bin",
    DOCS_FILE:     f"{R2_PUBLIC_URL}/documents.pkl",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Index file validation + download
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_lfs_pointer(path: str) -> bool:
    """
    Return True if the file is a Git LFS pointer (text) rather than a real
    binary.  LFS pointers start with b'version https://git-lfs'.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        return header.startswith(b"version https://")
    except OSError:
        return False


def ensure_indexes() -> None:
    """
    For each index file: if it is missing OR is a Git LFS pointer,
    download the real binary from Supabase Storage.
    """
    for local_path, url in INDEX_URLS.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        needs_download = (
            not os.path.exists(local_path)
            or _is_lfs_pointer(local_path)
        )

        if needs_download:
            filename = os.path.basename(local_path)
            print(f"  ⬇️  Downloading {filename} from Supabase Storage...")
            try:
                urllib.request.urlretrieve(url, local_path)
                size_mb = os.path.getsize(local_path) / 1_048_576
                print(f"  ✅  {filename}  ({size_mb:.1f} MB)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename} from {url}: {e}\n"
                    "Make sure the file is uploaded to Supabase Storage and the bucket is public."
                ) from e
        else:
            size_mb = os.path.getsize(local_path) / 1_048_576
            print(f"  ✅  {os.path.basename(local_path)} already present ({size_mb:.1f} MB)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="Monoko RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

rag: MonokoRAG | None = None


@app.on_event("startup")
def load_rag():
    global rag
    print("\n🚀 Starting Monoko RAG API...")
    try:
        print("  Checking index files...")
        ensure_indexes()
        rag = MonokoRAG()
        print("✅ RAG ready\n")
    except Exception as e:
        print(f"\n❌ Startup error: {e}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContextRequest(BaseModel):
    query: str
    top_k: int = 20


class ContextResponse(BaseModel):
    context: str
    query_lang: str
    result_count: int


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
        raise HTTPException(status_code=503, detail="RAG index not loaded.")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    query_lang = rag._detect_language(query)
    results = rag.retrieve(query, top_k=req.top_k)

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

    return ContextResponse(
        context="\n".join(parts),
        query_lang=query_lang,
        result_count=len(results),
    )
