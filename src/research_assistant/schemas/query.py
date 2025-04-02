from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    session_id: str # Require session ID for context
    llm_provider: Optional[str] = "deepseek"
    llm_model: Optional[str] = "deepseek-chat" # Specific model

class QueryResponse(BaseModel):
    session_id: str
    query: str
    response: str
    debug_info: Optional[dict] = None # For potential intermediate results