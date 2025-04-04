import chromadb
from research_assistant.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import logging
import uuid
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# --- Helper Function to Get Embedding Function ---
def get_embedding_function(provider: str):
    """Initializes and returns a Langchain embedding function based on the provider."""
    provider = provider.lower() if provider else "google" # Default to google if None
    if provider == "openai":
        if not settings.openai_api_key:
            logger.error("OpenAI API Key not found in settings for embeddings.")
            raise ValueError("OpenAI API Key missing for embeddings")
        logger.info("Initializing OpenAI Embeddings.")
        return OpenAIEmbeddings(
            model="text-embedding-3-small", # Or large, ada-002 etc.
            # dimensions=1024, # Optional: specify dimensions for V3 models
            openai_api_key=settings.openai_api_key
        )
    else:
        if not settings.google_api_key:
            logger.error("Google API Key not found in settings for embeddings.")
            raise ValueError("Google API Key missing for embeddings")
        logger.info("Initializing Google Generative AI Embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key
        )

class ChromaSessionStore:
    # Remove embedding_function initialization from __init__
    def __init__(self, path: str = settings.chroma_path):
        self.client = chromadb.PersistentClient(path=path)
        # No single self.embedding_function needed here anymore
        self._known_collections = set() # Simple cache for existing collections
        logger.info(f"ChromaDB client initialized at path: {path}")

    def _get_or_create_collection(self, session_id: str, embedding_provider_hint: str = "google"):
        """
        Gets a ChromaDB collection. If it doesn't exist, creates it
        using the specified embedding provider hint.
        """
        collection_name = f"session_{session_id}"

        try:
            collection = self.client.get_collection(name=collection_name)
            # If successful, cache it if not already cached
            self._known_collections.add(collection_name)
            logger.debug(f"Retrieved existing collection '{collection_name}'")
            return collection
        except Exception as e:
            # Handle specific Chroma exception if needed, e.g., DoesNotExistError
            # For simplicity, catching general Exception assumes it means "doesn't exist" or requires creation
            logger.info(f"Collection '{collection_name}' not found or error accessing ({type(e).__name__}), attempting creation with provider hint '{embedding_provider_hint}'.")

            try:
                # Determine provider and initialize the embedding function *only for creation*
                embedding_func = get_embedding_function(embedding_provider_hint)

                logger.info(f"Creating collection '{collection_name}' with {embedding_func.__class__.__name__}.")
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func, # Pass the function object
                    metadata={"hnsw:space": "cosine"} # Or other relevant metadata
                )
                self._known_collections.add(collection_name)
                logger.info(f"Successfully created collection '{collection_name}'")
                return collection
            except ValueError as ve: # Catch API key errors from get_embedding_function
                 logger.error(f"Failed to create collection {collection_name} due to configuration error: {ve}")
                 raise # Re-raise configuration errors
            except Exception as create_e:
                logger.error(f"Failed to create collection {collection_name}: {create_e}", exc_info=True)
                raise create_e # Re-raise unexpected creation errors


    # Modify add_message to accept the provider hint
    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history, using the specified embedding provider hint."""
        try:
            # Pass the hint to ensure collection is created with the correct embeddings
            collection = self._get_or_create_collection(session_id, embedding_provider_hint=embedding_provider)

            message_id = f"msg_{uuid.uuid4()}"
            collection.add(
                ids=[message_id],
                documents=[message.content],
                metadatas=[{"role": message.type, "timestamp": self._get_timestamp()}],
                # Embeddings are handled automatically by ChromaDB using the function
                # associated with the collection during the .add() call
            )
            logger.debug(f"Added message to session {session_id} (Provider hint: {embedding_provider})")
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}", exc_info=True)
            # Decide if we should raise or just log


    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history."""
        try:
            # Getting the collection doesn't need the hint once created
            # If it doesn't exist, _get_or_create will use the default ("google")
            # or potentially raise an error if creation fails.
            collection = self._get_or_create_collection(session_id)

            results = collection.get(
                include=["metadatas", "documents"],
            )
            # ... rest of history processing logic ...
            if not results or not results.get("ids"):
                 return []

            messages_data = sorted(
                [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(results["ids"], results["metadatas"], results["documents"])
                    if meta and 'timestamp' in meta
                ],
                key=lambda x: x["metadata"]["timestamp"],
                reverse=False
            )

            history: list[BaseMessage] = []
            # Limit processing to potentially relevant recent items
            for item in messages_data[-(limit*2):]:
                role = item["metadata"].get("role", "unknown")
                content = item["document"]
                if role == "human":
                    history.append(HumanMessage(content=content))
                elif role == "ai":
                    history.append(AIMessage(content=content))
                elif role == "system":
                     history.append(SystemMessage(content=content))

            logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
            return history[-limit:]

        except Exception as e:
            if "does not exist" in str(e).lower(): # Handle expected "not found" gracefully
                 logger.warning(f"Collection for session {session_id} not found. Returning empty history.")
                 return []
            logger.error(f"Error retrieving history for session {session_id}: {e}", exc_info=True)
            return [] # Return empty list on unexpected error


    def _get_timestamp(self) -> float:
        import time
        return time.time()

# Global instance (or use FastAPI dependency injection)
session_store = ChromaSessionStore()