from research_assistant.memory.chroma_store import ChromaSessionStore, session_store
from fastapi import HTTPException, status

# Dependency to get the global session store instance
def get_session_store() -> ChromaSessionStore:
    return session_store

# Example: Dependency to validate session ID (optional)
# async def validate_session_id(session_id: str) -> str:
#     # Basic validation logic (e.g., check format or existence if needed)
#     if not session_id or len(session_id) < 5: # Arbitrary length check
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Session ID format")
#     # You could potentially check if the session exists in ChromaDB here,
#     # but it might add latency. Often done within the endpoint logic.
#     return session_id