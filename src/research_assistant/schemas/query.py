from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    session_id: str # Require session ID for context
    llm_provider: str
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = "google"

class QueryResponse(BaseModel):
    session_id: str
    query: str
    response: str
    debug_info: Optional[dict] = None # For potential intermediate results