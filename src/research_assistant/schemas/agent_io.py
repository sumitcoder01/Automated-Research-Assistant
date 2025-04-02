from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class AgentOutput(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}