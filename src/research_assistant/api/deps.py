# src/research_assistant/deps.py (or main.py / config.py)
import os
import logging
from functools import lru_cache # Optional: Cache the store instance

# Import the base class if you have one
# from research_assistant.memory.base import BaseSessionStore
# Import specific implementations
from research_assistant.memory.chroma_store import ChromaSessionStore
from research_assistant.memory.pinecone_store import PineconeSessionStore
from research_assistant.config import settings

logger = logging.getLogger(__name__)

# Define BaseSessionStore if you didn't create a base.py
# This helps with type hinting, but isn't strictly necessary if you duck-type
from typing import Protocol, List
from langchain_core.messages import BaseMessage

class BaseSessionStore(Protocol):
    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"): ...
    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]: ...


# Use lru_cache(maxsize=1) for a simple singleton pattern
@lru_cache(maxsize=1)
def get_session_store() -> BaseSessionStore:
    """Gets the configured session store instance."""
    provider = settings.memory_provider.lower() # Default to pinecone
    logger.info(f"Initializing memory store with provider: {provider}")

    if provider == "pinecone":
        try:
            # Ensure Pinecone settings are available
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                 logger.error("MEMORY_PROVIDER is 'pinecone' but keys/env are missing in config.")
                 raise ValueError("Pinecone configuration missing for memory store.")
            return PineconeSessionStore()
        except (ValueError, ConnectionError) as e:
             logger.error(f"Failed to initialize PineconeSessionStore: {e}")
             raise # Stop application startup if primary store fails
    elif provider == "chroma":
         # Use Chroma for local dev or if specified
         logger.info(f"Using ChromaDB store at path: {settings.chroma_path}")
         # Ensure Chroma store init doesn't need embedding func now
         return ChromaSessionStore(path=settings.chroma_path)
    else:
        logger.error(f"Unsupported memory provider specified: {provider}")
        raise ValueError(f"Unsupported memory provider: {provider}")

# If you need a global instance (less flexible than Depends):
# try:
#     session_store = get_session_store()
# except Exception as e:
#     logger.critical(f"CRITICAL: Failed to initialize session store on startup: {e}")
#     session_store = None # Or exit

# Ensure FastAPI uses this dependency in endpoints:
# from research_assistant.api.deps import get_session_store
# async def handle_query(..., store: BaseSessionStore = Depends(get_session_store)): ...