# src/research_assistant/api/endpoints/session.py
from fastapi import APIRouter, Depends, HTTPException, status
import uuid
import logging
from research_assistant.schemas.session import SessionCreateRequest, SessionResponse, SessionHistory
from src.research_assistant.api.deps import get_session_store, BaseSessionStore

# Assuming router prefix is set in main.py (e.g., /api/v1/sessions)
router = APIRouter(tags=["Session Management"]) 
logger = logging.getLogger(__name__)

# Use "" for path relative to prefix in main.py
@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    store: BaseSessionStore = Depends(get_session_store) # store instance is available if needed
):
    """
    Creates a new research session identifier.
    The actual storage context (e.g., collection/namespace) is typically
    created lazily when the first message is added for this session_id.
    """
    session_id = request.session_id or f"sid_{uuid.uuid4()}"
    try:
        logger.info(f"Session identifier created: {session_id}")
        return SessionResponse(session_id=session_id, message="Session identifier created successfully")
    except Exception as e:
        # Catch potential errors if you add store interactions back later
        logger.error(f"Error during session creation for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initialize session: {e}")

# Use path relative to prefix in main.py
@router.get("/{session_id}/history", response_model=SessionHistory)
async def get_session_history(
    session_id: str,
    limit: int = 20, # How many messages to retrieve
    store: BaseSessionStore = Depends(get_session_store)
):
    """Retrieves the message history for a given session using the configured store."""
    logger.debug(f"Attempting to get history for session: {session_id}")
    try:
        # Call the get_history method on whatever store instance was injected
        history_messages = store.get_history(session_id, limit=limit)

        # Convert BaseMessage objects to dicts for JSON response
        history_dicts = []
        if history_messages: # Ensure it's not None or empty before iterating
             history_dicts = [{"role": msg.type, "content": msg.content} for msg in history_messages]

        logger.debug(f"Returning {len(history_dicts)} history messages for session: {session_id}")
        return SessionHistory(session_id=session_id, history=history_dicts)
    except Exception as e:
        # Catch potential errors from the specific store's get_history implementation
        logger.error(f"Could not retrieve history for session {session_id}: {e}", exc_info=True)
        # Consider specific error types if the store raises them (e.g., NotFoundError)
        # For now, generic 500, but 404 might be suitable if store indicates "not found" clearly
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving session history: {e}")