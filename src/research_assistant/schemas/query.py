from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class Citation(BaseModel):
    source: str
    url: str
    snippet: str

class Finding(BaseModel):
    category: str
    content: str
    citations: List[Citation]

class QueryResponse(BaseModel):
    query_id: str
    summary: str
    findings: List[Finding]
    citations: List[Citation]
    created_at: datetime = datetime.utcnow() 