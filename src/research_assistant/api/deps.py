# src/research_assistant/api/deps.py
import logging
from functools import lru_cache
from typing import Union # Add Union type hint

# Import the concrete implementations and base protocol
from research_assistant.memory.chroma_store import ChromaSessionStore
from research_assistant.memory.pinecone_store import PineconeSessionStore
from research_assistant.memory.base import BaseSessionStore
from research_assistant.config import settings

# Import low-level clients/indexes
import chromadb
from pinecone import Pinecone, Index # Import Pinecone and Index

logger = logging.getLogger(__name__)

# Existing dependency
@lru_cache(maxsize=1)
def get_session_store() -> BaseSessionStore:
    """Gets the configured session store instance based on settings.memory_provider."""
    provider = settings.memory_provider.lower()
    logger.info(f"Attempting to initialize memory store with provider: {provider}")

    if provider == "pinecone":
        try:
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                 logger.error("MEMORY_PROVIDER is 'pinecone' but PINECONE_API_KEY or PINECONE_ENVIRONMENT are missing.")
                 raise ValueError("Pinecone configuration missing for memory store.")
            logger.info("Initializing PineconeSessionStore...")
            store = PineconeSessionStore()
            logger.info("PineconeSessionStore initialized successfully.")
            return store
        except (ValueError, ConnectionError, TypeError, Exception) as e:
             logger.error(f"Failed to initialize PineconeSessionStore: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Pinecone session store: {e}")

    elif provider == "chroma":
         logger.info(f"Initializing ChromaSessionStore at path: {settings.chroma_path}")
         try:
             store = ChromaSessionStore(path=settings.chroma_path)
             logger.info("ChromaSessionStore initialized successfully.")
             return store
         except Exception as e:
             logger.error(f"Failed to initialize ChromaSessionStore: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Chroma session store: {e}")

    else:
        logger.error(f"Unsupported memory provider specified: {provider}")
        raise ValueError(f"Unsupported memory provider: {provider}")


# NEW Dependency for low-level backend access
@lru_cache(maxsize=1)
def get_vector_store_backend() -> Union[chromadb.Client, Index]:
    """
    Gets the low-level vector store client (Chroma) or index (Pinecone) object
    based on the configured memory_provider. Used for direct operations like
    document chunk storage.
    """
    provider = settings.memory_provider.lower()
    logger.info(f"Attempting to get low-level vector store backend for provider: {provider}")

    if provider == "pinecone":
        try:
            # Ensure Pinecone settings are available
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                 logger.error("MEMORY_PROVIDER is 'pinecone' but keys/env are missing in config.")
                 raise ValueError("Pinecone configuration missing for vector store backend.")
            if not settings.pinecone_index_name:
                 logger.error("MEMORY_PROVIDER is 'pinecone' but PINECONE_INDEX_NAME is missing.")
                 raise ValueError("Pinecone index name configuration missing.")

            # Initialize Pinecone client
            pc = Pinecone(api_key=settings.pinecone_api_key)

            # Check if index exists
            index_name = settings.pinecone_index_name
            if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
                 logger.error(f"Pinecone index '{index_name}' specified in settings does not exist.")
                 raise ValueError(f"Pinecone index '{index_name}' not found.")

            # Return the Index object
            index = pc.Index(index_name)
            logger.info(f"Returning Pinecone Index object for '{index_name}'.")
            # Perform a quick operation to ensure connectivity (optional but recommended)
            index.describe_index_stats()
            return index

        except (ValueError, ConnectionError, Exception) as e:
             logger.error(f"Failed to initialize Pinecone Index for vector store backend: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Pinecone vector store backend: {e}")

    elif provider == "chroma":
         logger.info(f"Returning ChromaDB client from path: {settings.chroma_path}")
         try:
             # Return the ChromaDB client instance
             client = chromadb.PersistentClient(path=settings.chroma_path)
             # Perform a quick operation to ensure connectivity (optional but recommended)
             client.heartbeat()
             logger.info(f"ChromaDB client backend initialized successfully at path: {settings.chroma_path}")
             return client
         except Exception as e:
             logger.error(f"Failed to initialize ChromaDB client for vector store backend: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Chroma vector store backend: {e}")

    else:
        logger.error(f"Unsupported memory provider specified for vector store backend: {provider}")
        raise ValueError(f"Unsupported memory provider for vector store backend: {provider}")
