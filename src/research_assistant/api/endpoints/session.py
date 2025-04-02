 from fastapi import APIRouter, Depends, HTTPException, status
from research_assistant.schemas.session import SessionCreateRequest, SessionResponse, SessionHistory
from research_assistant.memory.chroma_store import ChromaSessionStore
from research_assistant.api.deps import get_session_store
import uuid
import logging

router = APIRouter(
    prefix="/sessions",
    tags=["Session Management"]
)
logger = logging.getLogger(__name__)

@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    store: ChromaSessionStore = Depends(get_session_store)
):
    """Creates a new research session."""
    session_id = request.session_id or f"sid_{uuid.uuid4()}"
    try:
        # Ensure collection is created (or access is attempted)
        store._get_or_create_collection(session_id)
        logger.info(f"Session created successfully: {session_id}")
        return SessionResponse(session_id=session_id, message="Session created successfully")
    except Exception as e:
        logger.error(f"Failed to create session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create session: {e}")

@router.get("/{session_id}/history", response_model=SessionHistory)
async def get_session_history(
    session_id: str,
    limit: int = 20, # How many messages to retrieve
    store: ChromaSessionStore = Depends(get_session_store)
):
    """Retrieves the message history for a given session."""
    try:
        history_messages = store.get_history(session_id, limit=limit)
        # Convert BaseMessage objects to dicts for JSON response
        history_dicts = [{"role": msg.type, "content": msg.content} for msg in history_messages]
        return SessionHistory(session_id=session_id, history=history_dicts)
    except Exception as e:
        # Handle case where session might not exist
        logger.warning(f"Could not retrieve history for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session not found or error retrieving history: {e}")