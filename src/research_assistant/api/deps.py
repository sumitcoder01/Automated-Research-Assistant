import os
import logging
from functools import lru_cache
from research_assistant.memory.chroma_store import ChromaSessionStore
from research_assistant.memory.pinecone_store import PineconeSessionStore
from research_assistant.config import settings

logger = logging.getLogger(__name__)

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