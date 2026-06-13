import os
import shutil
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from agent.agent_controller import agent_controller
from mcp.protocol import MCPMessage, MCPResponse, ToolStatus
from mcp.communication import mcp_bus
from mcp.tool_registry import tool_registry
from retrieval.vector_store import vector_store
from llm.ollama_client import ollama_client
from tools.ieee_compliance_store import get_compliance, get_all_compliance

router = APIRouter(prefix="/api/v1")

# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query query for the AI agent")
    top_k: int = Field(default=5, description="Number of source documents to return")
    threshold: float = Field(default=0.0, description="Similarity matching cutoff (default 0.0)")

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    plan: List[str]
    tool_trace: List[Dict[str, Any]]
    sources: List[str]
    chunks: List[Dict[str, Any]]
    confidence: float
    num_chunks_retrieved: int
    elapsed_ms: float
    compliance_report: Optional[Dict[str, Any]] = None

# 1. GET /health
@router.get("/health", response_model=Dict[str, Any])
async def get_health() -> Dict[str, Any]:
    """Check core status of system database and local LLM presence."""
    llm_status = await ollama_client.get_status()
    chunk_count = vector_store.get_chunk_count()
    
    status_color = "green"
    if not llm_status["online"]:
        status_color = "red"
    elif not llm_status["configured_model_available"]:
        status_color = "yellow"
        
    return {
        "status": status_color,
        "llm_online": llm_status["online"],
        "llm_model": settings.OLLAMA_MODEL,
        "configured_model_available": llm_status["configured_model_available"],
        "chunk_count": chunk_count
    }

# 2. GET /system/status
@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status() -> Dict[str, Any]:
    """Full architectural telemetry summary for dashboard indicators."""
    llm_status = await ollama_client.get_status()
    chunk_count = vector_store.get_chunk_count()
    documents = vector_store.get_documents()
    registered_tools = tool_registry.get_registered_tools()
    
    return {
        "status": "online",
        "llm": llm_status,
        "vector_store": {
            "chunk_count": chunk_count,
            "document_count": len(documents),
            "documents": documents
        },
        "mcp_tools": registered_tools
    }

# 3. POST /query
@router.post("/query", response_model=QueryResponse)
async def post_query(request: QueryRequest) -> QueryResponse:
    try:
        response_payload = await agent_controller.handle_query(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )

        # Agent controller already attached the asynchronous compliance_report
        return QueryResponse(**response_payload)
    except Exception as e:
        logger.exception("FastAPI POST /query endpoint error")
        raise HTTPException(
            status_code=500, 
            detail=f"Agent process failed: {str(e)}"
        )

# 4. POST /documents/upload
@router.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload document, write to disk, then trigger upload_document MCP ingest."""
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{ext}' is not supported. Allowed formats: {settings.ALLOWED_EXTENSIONS}"
        )
        
    doc_id = str(uuid.uuid4())
    temp_file_path = settings.UPLOADS_DIR / f"{doc_id}{ext}"
    
    try:
        # Save file to uploads directory
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File uploaded to {temp_file_path}. Initiating MCP ingest tool...")
        
        # Build MCPMessage to run in-memory ingest
        mcp_msg = MCPMessage(
            tool_name="upload_document",
            action="ingest",
            payload={
                "file_path": str(temp_file_path),
                "filename": filename,
                "doc_id": doc_id
            }
        )
        
        response: MCPResponse = await mcp_bus.send(mcp_msg)
        
        if response.status == ToolStatus.ERROR:
            # CRITICAL: Read the error using .error_detail
            raise HTTPException(
                status_code=500, 
                detail=f"Ingestion failed: {response.error_detail}"
            )
            
        return {
            "success": True,
            "message": "Document indexed successfully",
            "data": response.data
        }
        
    except Exception as e:
        # Clean up temp file in case of exception
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logger.exception("Upload endpoint failed")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Document upload/parsing error: {str(e)}")

# 5. GET /documents
@router.get("/documents", response_model=List[Dict[str, Any]])
async def get_documents() -> List[Dict[str, Any]]:
    """Return all unique documents currently indexed in vector store."""
    return vector_store.get_documents()

# 6. DELETE /documents/{doc_id}
@router.delete("/documents/{doc_id}", response_model=Dict[str, Any])
async def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete document by doc_id and re-build FAISS index."""
    try:
        mcp_msg = MCPMessage(
            tool_name="upload_document",
            action="delete",
            payload={"doc_id": doc_id}
        )
        
        response: MCPResponse = await mcp_bus.send(mcp_msg)
        
        if response.status == ToolStatus.ERROR:
            # CRITICAL: Read the error using .error_detail
            raise HTTPException(
                status_code=500, 
                detail=f"Purge failed: {response.error_detail}"
            )
            
        return {
            "success": True,
            "message": f"Document {doc_id} purged and vector database successfully re-indexed."
        }
    except Exception as e:
        logger.exception(f"Purge request failed for doc_id {doc_id}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# 7. GET /mcp/tools
@router.get("/mcp/tools", response_model=List[Dict[str, Any]])
async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get all registered tools inside SIRS MCP Bus registry."""
    return tool_registry.get_registered_tools()

# 8. GET /mcp/log
@router.get("/mcp/log", response_model=List[Dict[str, Any]])
async def get_mcp_log() -> List[Dict[str, Any]]:
    """Fetch live audit trail logs of all tool requests handled by MCPBus."""
    return mcp_bus.get_audit_log()

# 9. GET /settings
@router.get("/settings", response_model=Dict[str, Any])
async def get_settings() -> Dict[str, Any]:
    """Expose application config parameters for system settings overview."""
    llm_status = await ollama_client.get_status()
    
    return {
        "SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD,
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "TOP_K": settings.TOP_K,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "EMBEDDING_DIM": settings.EMBEDDING_DIM,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "LLM_TEMPERATURE": settings.LLM_TEMPERATURE,
        "LLM_MAX_TOKENS": settings.LLM_MAX_TOKENS,
        "MAX_UPLOAD_SIZE_MB": settings.MAX_UPLOAD_SIZE_MB,
        "ALLOWED_EXTENSIONS": settings.ALLOWED_EXTENSIONS,
        "ollama_models": llm_status.get("models", [])
    }

# 10. GET /documents/{doc_id}/compliance
@router.get("/documents/{doc_id}/compliance", response_model=Dict[str, Any])
async def get_document_compliance(doc_id: str) -> Dict[str, Any]:
    """
    Get the full IEEE compliance report for a specific uploaded document.
    The report is generated at upload time by scanning the document text.
    """
    from tools.ieee_compliance_store import get_compliance
    result = get_compliance(doc_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No IEEE compliance data found for document '{doc_id}'. "
            f"Re-upload the document to generate a compliance report."
        )
    return result

# 11. GET /documents/compliance/all
@router.get("/documents/compliance/all", response_model=Dict[str, Any])
async def get_all_documents_compliance() -> Dict[str, Any]:
    """
    Get IEEE compliance summaries for ALL uploaded documents.
    Returns a dict keyed by doc_id with score and verdict per document.
    """
    from tools.ieee_compliance_store import get_all_compliance
    all_data = get_all_compliance()
 
    # Return lightweight summary (not full report) for the document list view
    summary = {}
    for doc_id, entry in all_data.items():
        report = entry.get("report", {})
        summary[doc_id] = {
            "filename":          entry.get("filename"),
            "overall_score_pct": report.get("overall_score_pct", 0),
            "overall_passed":    report.get("overall_passed", False),
            "verdict":           report.get("verdict", "Unknown"),
            "standards": [
                {
                    "id":       s.get("id"),
                    "passed":   s.get("passed"),
                    "score_pct": s.get("score_pct"),
                }
                for s in report.get("standards", [])
            ]
        }
    return summary