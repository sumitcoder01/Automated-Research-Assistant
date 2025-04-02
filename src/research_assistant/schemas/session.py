from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class SessionCreate(BaseModel):
    name: str
    created_at: datetime = datetime.utcnow()

class SessionResponse(BaseModel):
    session_id: str
    name: str
    created_at: datetime
    last_activity: Optional[datetime] = None
    query_count: int = 0 