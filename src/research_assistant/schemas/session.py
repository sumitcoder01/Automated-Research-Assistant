from pydantic import BaseModel, Field
import uuid

class SessionCreateRequest(BaseModel):
    # Optionally allow client to suggest an ID, otherwise generate one
    session_id: str | None = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

class SessionHistory(BaseModel):
    session_id: str
    history: list[dict] # e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]