"""
MONOKO - Step 3: Build the RAG System
======================================
Creates a vector store from the merged Lingala knowledge base,
then provides a query interface for the Lingala AI agent.

This is the BRAIN of your AI dialect speaker.

Requirements:
    pip install sentence-transformers faiss-cpu anthropic openai

Usage:
    python step3_build_rag.py             # Build the vector store
    python step3_build_rag.py --query     # Interactive query mode
    python step3_build_rag.py --test      # Run test queries
"""

import os
import json
import pickle
import argparse
import numpy as np
from typing import List, Dict, Tuple


DATA_DIR   = "monoko_data"
MERGED_DIR = os.path.join(DATA_DIR, "merged")
RAG_DIR    = os.path.join(DATA_DIR, "rag_index")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
os.makedirs(RAG_DIR, exist_ok=True)

KB_FILE      = os.path.join(MERGED_DIR, "lingala_knowledge_base.jsonl")
NLLB_HC_FILE = os.path.join(PROC_DIR, "nllb_ngram_filtered.jsonl")
INDEX_FR_FILE = os.path.join(RAG_DIR, "faiss_index_fr.bin")   # French-side embeddings
INDEX_LN_FILE = os.path.join(RAG_DIR, "faiss_index_ln.bin")   # Lingala-side embeddings
DOCS_FILE     = os.path.join(RAG_DIR, "documents.pkl")        # shared document list

# Only include NLLB entries whose character n-gram profile closely matches
# verified Lingala (output of score_nllb_ngram.py).
NLLB_NGRAM_THRESHOLD = -6.0

# Hybrid retrieval constants
_HIGH_QUALITY = {"verified", "gold"}
_POOL_SIZE    = 300   # FAISS candidates fetched before quality partitioning
# NLLB re-ranking blend: rank = sim_score + VOCAB_BLEND * vocab_score
_VOCAB_BLEND  = 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3A: Build the Vector Store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_knowledge_base() -> List[Dict]:
    """
    Load the knowledge base.

    Priority tiers (loaded in order, highest quality first):
      1. verified  — Supabase professor-checked pairs
      2. gold      — FLORES-200 human translations
      3. nllb_hc   — High-confidence NLLB (ngram_score > NLLB_NGRAM_THRESHOLD,
                     cleaned by clean_nllb.py + score_nllb_ngram.py)
    """
    if not os.path.exists(KB_FILE):
        print(f"❌ Knowledge base not found: {KB_FILE}")
        print("   Run step2_process_and_merge.py first.")
        return []

    docs = []

    # Tier 1 & 2: verified + gold from merged KB
    with open(KB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("quality") in ("verified", "gold"):
                docs.append(doc)
    print(f"  verified + gold : {len(docs):,} entries")

    # Tier 3: high-confidence NLLB
    nllb_count = 0
    if os.path.exists(NLLB_HC_FILE):
        with open(NLLB_HC_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if doc.get("ngram_score", -99) > NLLB_NGRAM_THRESHOLD:
                    docs.append(doc)
                    nllb_count += 1
        print(f"  NLLB HC         : {nllb_count:,} entries  (ngram_score > {NLLB_NGRAM_THRESHOLD})")
    else:
        print(f"  NLLB HC file not found — skipping ({NLLB_HC_FILE})")

    print(f"✅ Total knowledge base: {len(docs):,} entries")
    return docs


def create_rag_documents(entries: List[Dict]) -> List[Dict]:
    """
    Convert knowledge base entries into RAG documents.
    Each document is a text chunk that can be retrieved and 
    used as context for the LLM.
    """
    documents = []
    
    for i, entry in enumerate(entries):
        fr = entry.get("french", "")
        ln = entry.get("lingala", "")
        source = entry.get("source", "unknown")
        quality = entry.get("quality", "auto")
        entry_type = entry.get("type", "sentence")
        
        # Create a rich text representation for embedding
        if entry_type == "word":
            text = f"Mot: {fr} → Lingala: {ln}"
        else:
            text = f"Français: {fr}\nLingala: {ln}"
        
        documents.append({
            "id":          i,
            "text":        text,
            "french":      fr,
            "lingala":     ln,
            "source":      source,
            "quality":     quality,
            "type":        entry_type,
            # Carried for NLLB re-ranking; 0.0 for verified/gold entries
            "vocab_score": float(entry.get("vocab_score", 0.0)),
        })
    
    return documents


def build_vector_index(documents: List[Dict]):
    """
    Build two FAISS indexes from the same document list:
      - French-side index  (faiss_index_fr.bin): used when user writes in French/English
      - Lingala-side index (faiss_index_ln.bin): used when user writes in Lingala

    Both indexes share the same documents.pkl so retrieved idx values map to
    the same document regardless of which index is queried.
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Missing packages. Run:")
        print("   pip install sentence-transformers faiss-cpu")
        return

    print("\n📐 Loading embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Model loaded")

    batch_size = 256

    # ── French index ──────────────────────────────────────────────────────────
    print(f"\n📊 [1/2] Encoding French side ({len(documents):,} documents)...")
    fr_embeddings = model.encode(
        [doc["french"] for doc in documents],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dim = fr_embeddings.shape[1]
    index_fr = faiss.IndexFlatIP(dim)
    index_fr.add(fr_embeddings.astype(np.float32))
    faiss.write_index(index_fr, INDEX_FR_FILE)
    print(f"✅ French index: {index_fr.ntotal:,} vectors  →  {INDEX_FR_FILE}")

    # ── Lingala index ─────────────────────────────────────────────────────────
    print(f"\n📊 [2/2] Encoding Lingala side ({len(documents):,} documents)...")
    ln_embeddings = model.encode(
        [doc["lingala"] for doc in documents],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    index_ln = faiss.IndexFlatIP(dim)
    index_ln.add(ln_embeddings.astype(np.float32))
    faiss.write_index(index_ln, INDEX_LN_FILE)
    print(f"✅ Lingala index: {index_ln.ntotal:,} vectors  →  {INDEX_LN_FILE}")

    # ── Shared document list ──────────────────────────────────────────────────
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)
    print(f"💾 Documents saved: {DOCS_FILE}")

    return index_fr, index_ln, documents, model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3B: Query the RAG System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MonokoRAG:
    """The Monoko Lingala RAG query engine."""

    # Languages routed to the French index (query is about French or English content)
    _FR_LANGS = {"fr", "en"}

    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        print("🔄 Loading RAG system...")

        missing = [p for p in (INDEX_FR_FILE, INDEX_LN_FILE, DOCS_FILE)
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"RAG index files missing: {missing}\n"
                "Run: python step3_build_rag.py"
            )

        self.index_fr = faiss.read_index(INDEX_FR_FILE)
        self.index_ln = faiss.read_index(INDEX_LN_FILE)
        with open(DOCS_FILE, "rb") as f:
            self.documents = pickle.load(f)

        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        print(f"✅ RAG loaded: {self.index_fr.ntotal:,} FR vectors, "
              f"{self.index_ln.ntotal:,} LN vectors, "
              f"{len(self.documents):,} documents")

    def _detect_language(self, text: str) -> str:
        """
        Returns 'fr' for French/English queries (→ French index)
        or 'ln' for Lingala queries (→ Lingala index).

        langdetect has no Lingala model — Lingala gets classified as Swahili,
        Tagalog, Indonesian, or similar. We treat anything that isn't
        confidently French or English as Lingala.
        """
        try:
            from langdetect import detect, LangDetectException
            lang = detect(text) if len(text.strip()) >= 8 else "fr"
            return "fr" if lang in self._FR_LANGS else "ln"
        except Exception:
            return "fr"

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Dual-path hybrid retrieval.

        Language detection routes the query to the right index:
          - French / English  →  French-side index  (translation requests)
          - Lingala / other   →  Lingala-side index  (conversation, grammar)

        Within each path: verified/gold entries fill first, NLLB fills
        remaining slots ranked by (sim_score + VOCAB_BLEND * vocab_score).

        Args:
            query: The user's message in any language
            top_k: Number of results to return

        Returns:
            List of relevant documents with similarity_score and query_lang fields
        """
        query_lang = self._detect_language(query)
        index = self.index_fr if query_lang == "fr" else self.index_ln
        pool_size = min(_POOL_SIZE, index.ntotal)

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = index.search(query_embedding, pool_size)

        high_quality = []
        auto = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(score)
                doc["query_lang"] = query_lang
                if doc.get("quality") in _HIGH_QUALITY:
                    high_quality.append(doc)
                else:
                    doc["_nllb_rank"] = (
                        float(score) + _VOCAB_BLEND * doc.get("vocab_score", 0.0)
                    )
                    auto.append(doc)

        high_quality.sort(key=lambda x: -x["similarity_score"])
        auto.sort(key=lambda x: -x["_nllb_rank"])

        remaining = max(0, top_k - len(high_quality))
        return (high_quality + auto[:remaining])[:top_k]
    
    def build_context(self, query: str, max_context_pairs: int = 30) -> str:
        """
        Build the context string to inject into the LLM prompt.
        This is the RAG part — retrieving relevant knowledge.
        """
        results = self.retrieve(query, top_k=max_context_pairs)
        
        context_parts = []
        
        # Separate words and sentences
        words = [r for r in results if r.get("type") == "word"]
        sentences = [r for r in results if r.get("type") != "word"]
        
        if words:
            context_parts.append("=== VOCABULAIRE ===")
            for w in words[:10]:
                context_parts.append(f"• {w['french']} → {w['lingala']}")
        
        if sentences:
            context_parts.append("\n=== PHRASES PARALLÈLES ===")
            for s in sentences[:20]:
                q_tag = f" [{s['quality']}]" if s.get('quality') == 'verified' else ""
                context_parts.append(f"FR: {s['french']}")
                context_parts.append(f"LN: {s['lingala']}{q_tag}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def query_llm(self, user_message: str, provider: str = "anthropic") -> str:
        """
        Full RAG pipeline: retrieve context → build prompt → call LLM.
        
        Args:
            user_message: What the user said/asked
            provider: "anthropic" (Claude) or "openai" (GPT-4)
        
        Returns:
            The AI's response in Lingala (with French explanation)
        """
        # Step 1: Retrieve relevant context
        context = self.build_context(user_message)
        
        # Step 2: Build the system prompt
        system_prompt = f"""Tu es Monoko, un assistant IA qui parle Lingala couramment. 
Tu es le premier agent conversationnel au monde spécialisé dans la langue Lingala.

Ton rôle:
- Répondre aux questions en Lingala quand l'utilisateur parle Lingala
- Traduire du français vers le Lingala et vice versa
- Enseigner le Lingala aux apprenants
- Tenir des conversations naturelles en Lingala
- Expliquer la grammaire et le vocabulaire Lingala

Règles importantes:
1. Quand l'utilisateur parle en Lingala, réponds en Lingala puis donne la traduction française entre parenthèses.
2. Quand l'utilisateur demande une traduction, fournis la traduction avec une explication.
3. Utilise les données du corpus ci-dessous comme référence pour tes traductions. 
   Les entrées marquées [verified] sont vérifiées par des professeurs — privilégie-les.
4. Si tu n'es pas sûr d'une traduction, dis-le honnêtement plutôt que d'inventer.
5. Utilise un ton chaleureux et encourageant pour motiver l'apprentissage.

=== CORPUS DE RÉFÉRENCE (Français ↔ Lingala) ===
{context}
=== FIN DU CORPUS ===

Réponds de manière naturelle en utilisant ce corpus comme base de connaissances.
Si la question porte sur du vocabulaire ou des phrases présents dans le corpus, utilise-les directement.
"""
        
        # Step 3: Call the LLM
        if provider == "anthropic":
            return self._call_anthropic(system_prompt, user_message)
        elif provider == "openai":
            return self._call_openai(system_prompt, user_message)
        else:
            return f"[Provider '{provider}' not supported. Use 'anthropic' or 'openai']"
    
    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
            
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            return response.content[0].text
        except ImportError:
            return "❌ Install anthropic: pip install anthropic"
        except Exception as e:
            return f"❌ Anthropic API error: {e}"
    
    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI()  # Uses OPENAI_API_KEY env var
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.choices[0].message.content
        except ImportError:
            return "❌ Install openai: pip install openai"
        except Exception as e:
            return f"❌ OpenAI API error: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERACTIVE MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def interactive_mode(provider="anthropic"):
    """Run the RAG system in interactive chat mode."""
    
    print("\n" + "━" * 60)
    print("  🗣️  MONOKO - Lingala AI Agent")
    print("  Type in French or Lingala. Type 'quit' to exit.")
    print("━" * 60 + "\n")
    
    rag = MonokoRAG()
    
    while True:
        try:
            user_input = input("\n🧑 Toi: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Kende malamu! (Au revoir!)\n")
            break
        
        print("\n🤖 Monoko: ", end="", flush=True)
        response = rag.query_llm(user_input, provider=provider)
        print(response)


def test_mode():
    """Run test queries to validate both retrieval paths."""

    print("\n" + "━" * 60)
    print("  🧪 MONOKO - RAG Test Mode (dual-path retrieval)")
    print("━" * 60 + "\n")

    rag = MonokoRAG()

    # French/English queries → French index
    # Lingala queries        → Lingala index
    test_queries = [
        # ── French path ───────────────────────────────────────────────────────
        "Comment dit-on bonjour en Lingala?",
        "Traduis 'je vais au marché' en Lingala",
        "Quels sont les pronoms en Lingala?",
        "Comment on compte de 1 à 10 en Lingala?",
        # ── Lingala path ──────────────────────────────────────────────────────
        "Mbote na yo, ozali malamu?",
        "Nakei na marché lelo",
        "Ndenge nini ko salela verbe na Lingala?",
        "Biso tozali bato ya Kinshasa",
    ]

    for query in test_queries:
        results = rag.retrieve(query, top_k=20)
        lang = results[0].get("query_lang", "?") if results else "?"
        index_used = "🇫🇷 FR index" if lang == "fr" else "🇨🇩 LN index"

        print(f"\n📝 [{index_used}] {query}")
        print("─" * 50)
        for r in results[:3]:
            print(f"   • [{r['quality']:8}] sim={r['similarity_score']:.3f}"
                  f"  FR: {r['french'][:40]:40}  LN: {r['lingala'][:35]}")
        print("─" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monoko Lingala RAG System")
    parser.add_argument("--query", action="store_true", help="Interactive query mode")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    args = parser.parse_args()
    
    if args.query:
        interactive_mode(provider=args.provider)
    elif args.test:
        test_mode()
    else:
        # Build mode (default)
        print("\n" + "━" * 60)
        print("  MONOKO - Building RAG Vector Store")
        print("━" * 60 + "\n")
        
        entries = load_knowledge_base()
        if entries:
            documents = create_rag_documents(entries)
            build_vector_index(documents)  # builds both FR + LN indexes
            
            print(f"\n✅ RAG system ready!")
            print(f"   Run: python step3_build_rag.py --query")
            print(f"   Or:  python step3_build_rag.py --test\n")
