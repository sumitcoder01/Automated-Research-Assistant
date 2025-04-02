from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from .orchestrator import ResearchOrchestrator
from .memory import MemoryStore
from .config import settings

app = FastAPI(
    title="Automated Research Assistant",
    description="A multi-agent system for automated research and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = ResearchOrchestrator()
memory = MemoryStore()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    results: List[Dict[str, Any]]

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a research query."""
    try:
        result = await orchestrator.process_query(
            query=request.query,
            session_id=request.session_id,
            llm_provider=request.llm_provider,
            model=request.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session by ID."""
    session = await memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 10):
    """Get the interaction history for a session."""
    history = await memory.get_session_history(session_id, limit)
    return history

@app.get("/api/sessions/search")
async def search_sessions(query: str, limit: int = 5):
    """Search for sessions based on their interaction history."""
    sessions = await memory.search_sessions(query, limit)
    return sessions

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    ) 