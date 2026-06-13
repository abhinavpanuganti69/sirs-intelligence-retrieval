# ==============================================================================
# SIRS — Standards Indexer
# File: backend/standards_indexer.py
#
# PURPOSE: One-time script to ingest IEEE standard PDFs into a dedicated
#          FAISS index (standards_index/). Run this ONCE after placing PDFs
#          in the backend/standards/ folder.
#
# USAGE (from backend/ directory, with venv activated):
#   python standards_indexer.py
#
# OUTPUT:
#   backend/data/standards_index/standards.index       ← FAISS binary index
#   backend/data/standards_index/standards_metadata.json ← chunk metadata
# ==============================================================================

import os
import json
import sys
import numpy as np

# ── Path setup (run from backend/) ──────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
STANDARDS_DIR  = os.path.join(BASE_DIR, "standards")
INDEX_DIR      = os.path.join(BASE_DIR, "data", "standards_index")

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # same model as existing vector_store.py
CHUNK_SIZE      = 450                   # characters — captures individual clauses
CHUNK_OVERLAP   = 80                    # overlap so clauses don't get split cold

# Map: filename → standard ID used in compliance reports
STANDARDS_MAP = {
    "IEEE_830.pdf":  "IEEE 830",
    "IEEE_829.pdf":  "IEEE 829",
    "IEEE_1016.pdf": "IEEE 1016",
}


# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Returns list of {page: int, text: str}
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("[ERROR] PyPDF2 not found. Run: pip install PyPDF2")
        sys.exit(1)

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE characters.
    Overlap ensures clause boundaries are not lost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:          # skip tiny leftover fragments
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Main Indexer ──────────────────────────────────────────────────────────────
def build_standards_index():
    try:
        import faiss
    except ImportError:
        print("[ERROR] faiss-cpu not found. Run: pip install faiss-cpu")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not found. Run: pip install sentence-transformers")
        sys.exit(1)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── Validate standards folder ────────────────────────────────────────────
    if not os.path.exists(STANDARDS_DIR):
        print(f"[ERROR] standards/ folder not found at: {STANDARDS_DIR}")
        print("        Create backend/standards/ and place your IEEE PDFs there.")
        sys.exit(1)

    # ── Load embedding model ─────────────────────────────────────────────────
    print(f"\n[SIRS] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[SIRS] Model loaded ✓")

    all_chunks: list[dict] = []
    all_embeddings: list  = []
    found_any = False

    # ── Process each PDF ─────────────────────────────────────────────────────
    for filename, standard_id in STANDARDS_MAP.items():
        pdf_path = os.path.join(STANDARDS_DIR, filename)

        if not os.path.exists(pdf_path):
            print(f"\n[WARN] {filename} not found in standards/ — skipping {standard_id}")
            continue

        found_any = True
        print(f"\n[SIRS] Processing {standard_id}  ({filename}) ...")

        pages = extract_pages(pdf_path)
        if not pages:
            print(f"  [WARN] Could not extract any text from {filename}. Skipping.")
            continue

        # Build chunks for this standard
        doc_chunks = []
        for page_data in pages:
            text_chunks = chunk_text(page_data["text"])
            for idx, chunk in enumerate(text_chunks):
                doc_chunks.append({
                    "text":        chunk,
                    "standard_id": standard_id,
                    "page":        page_data["page"],
                    "source":      filename,
                    "chunk_index": idx,
                })

        print(f"  → Pages extracted : {len(pages)}")
        print(f"  → Chunks generated: {len(doc_chunks)}")

        # Embed chunks in batches
        texts = [c["text"] for c in doc_chunks]
        print(f"  → Embedding {len(texts)} chunks (batch_size=32) ...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        all_chunks.extend(doc_chunks)
        all_embeddings.extend(embeddings)
        print(f"  → {standard_id} done ✓")

    if not found_any:
        print("\n[ERROR] No PDFs found in standards/ folder.")
        print("        Place IEEE_830.pdf, IEEE_829.pdf, IEEE_1016.pdf there and retry.")
        sys.exit(1)

    if not all_chunks:
        print("\n[ERROR] No chunks were generated. PDFs may be scanned images (not text-based).")
        sys.exit(1)

    # ── Build FAISS index ────────────────────────────────────────────────────
    print(f"\n[SIRS] Building FAISS index ...")
    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)

    # Save FAISS index
    faiss_path = os.path.join(INDEX_DIR, "standards.index")
    faiss.write_index(index, faiss_path)
    print(f"  → FAISS index saved  : {faiss_path}")

    # Save metadata (chunk text + source info for every vector)
    metadata_path = os.path.join(INDEX_DIR, "standards_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"  → Metadata saved     : {metadata_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    indexed_standards = list({c["standard_id"] for c in all_chunks})
    print(f"""
╔══════════════════════════════════════════════════════╗
║        Standards Index Built Successfully ✅          ║
╠══════════════════════════════════════════════════════╣
║  Standards indexed : {str(indexed_standards):<33}║
║  Total chunks      : {len(all_chunks):<33}║
║  Vector dimension  : {dimension:<33}║
║  Index location    : data/standards_index/           ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    build_standards_index()