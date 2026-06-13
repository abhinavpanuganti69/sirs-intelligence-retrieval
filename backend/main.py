from contextlib import asynccontextmanager
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from retrieval.embeddings import embedding_model
from retrieval.vector_store import vector_store
from api.routes import router
from routes.ieee_compliance_route import router as compliance_router
from retrieval.standards_vector_store import get_standards_store

# Import tools explicitly to trigger their registration decorators
import tools.retrieval_tool
import tools.summarizer_tool
import tools.upload_tool
import tools.report_tool
import tools.list_tool
from mcp.tool_registry import tool_registry

# Configure loguru logging output format and file log location
logger.add(
    settings.LOGS_DIR / "sirs_backend.log", 
    rotation="10 MB", 
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    SIRS Backend Lifespan:
    1. Pre-load the sentence-transformers model (all-MiniLM-L6-v2) for embeddings.
    2. Warmup FAISS Vector store and validate index/metadata sync.
    3. Warmup IEEE Standards Vector store.
    4. Log active registered tools from registry.
    """
    logger.info("Initializing SIRS Lifespan Sequence...")
    
    # 1. Warm up embedding model
    start_embed = time.perf_counter()
    embedding_model.load()
    logger.info(f"Embedding model ready in {(time.perf_counter() - start_embed):.2f}s")
    
    # 2. Vector Store Initialization
    start_vs = time.perf_counter()
    count = vector_store.get_chunk_count()
    logger.info(f"FAISS Database loaded successfully with {count} active chunks in {(time.perf_counter() - start_vs):.2f}s")
    
    # ── NEW: Warm up IEEE Standards Vector Store ──
    standards_store = get_standards_store()
    logger.info(f"Standards index ready: {standards_store.stats()}")
    # ──────────────────────────────────────────────
    
    # 3. Log registered MCP Tools list
    tools = tool_registry.get_registered_tools()
    logger.info(f"Verified {len(tools)} active registered MCP tools:")
    for t in tools:
        logger.info(f" - Tool: {t['name']} ({t['description']})")
        
    yield
    
    logger.info("SIRS Backend shutting down...")

app = FastAPI(
    title="Secure Offline Intelligence Retrieval System (SIRS)",
    description="A secure, offline-first RAG and Agentic MCP system for defense intelligence operations.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for the frontend (Vite React development on http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes API router
app.include_router(router)
app.include_router(compliance_router, prefix="/compliance", tags=["IEEE Compliance"])

@app.get("/")
def read_root():
    return {
        "system": "Secure Offline Intelligence Retrieval System (SIRS)",
        "version": "1.0.0",
        "classification": "UNCLASSIFIED // DEVELOPMENT"
    }