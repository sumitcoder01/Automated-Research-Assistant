# src/research_assistant/memory/chroma_store.py
import chromadb
import logging
import uuid
import time # For _get_timestamp
from typing import List
from research_assistant.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
# Import the base class and the shared embedding helper
from research_assistant.memory.base import BaseSessionStore
from research_assistant.utils.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

class ChromaSessionStore(BaseSessionStore): # Implement the protocol
    def __init__(self, path: str = settings.chroma_path):
        self.client = chromadb.PersistentClient(path=path)
        self._known_collections = set()
        logger.info(f"ChromaDB client initialized at path: {path}")

    def _get_or_create_collection(self, session_id: str, embedding_provider_hint: str = "google"):
        collection_name = f"session_{session_id}"
        if collection_name in self._known_collections:
             try:
                 # Still quickly verify it exists in case of external deletion/corruption
                 return self.client.get_collection(name=collection_name)
             except Exception as get_e:
                 logger.warning(f"Collection '{collection_name}' was known but failed on get: {get_e}. Attempting recreation.")
                 self._known_collections.remove(collection_name) # Remove from cache

        try:
            # Try getting first in case it exists but isn't cached
            collection = self.client.get_collection(name=collection_name)
            self._known_collections.add(collection_name)
            logger.debug(f"Retrieved existing collection '{collection_name}'")
            return collection
        except Exception: # Broad exception, assumes collection doesn't exist or isn't accessible
            logger.info(f"Collection '{collection_name}' not found or error accessing, attempting creation with provider hint '{embedding_provider_hint}'.")
            try:
                # Use the shared embedding function helper
                embedding_func = get_embedding_function(embedding_provider_hint)
                # Pass the Langchain embedding function *object* to ChromaDB
                # Ensure the chromadb library version supports passing Langchain functions directly
                # If not, you might need chromadb.utils.embedding_functions.LangchainEmbeddingFunction(embedding_func)
                collection = self.client.create_collection(
                    name=collection_name,
                    # Pass the embedding function object itself if Chroma supports it
                    # If not, you might need a Chroma-specific wrapper
                    embedding_function=chromadb.utils.embedding_functions.LangchainEmbeddingFunction(embedding_func), # Example using Chroma's wrapper
                    metadata={"hnsw:space": "cosine"}
                )
                self._known_collections.add(collection_name)
                logger.info(f"Successfully created collection '{collection_name}' using {embedding_func.__class__.__name__}")
                return collection
            except ValueError as ve:
                 logger.error(f"Failed to create collection {collection_name} due to configuration error: {ve}")
                 raise
            except Exception as create_e:
                logger.error(f"Failed to create collection {collection_name}: {create_e}", exc_info=True)
                raise create_e

    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history, using the specified embedding provider hint."""
        try:
            # Ensure content is not None or empty
            if not message.content:
                logger.warning(f"Attempted to add message with empty content to session {session_id}. Skipping.")
                return

            collection = self._get_or_create_collection(session_id, embedding_provider_hint=embedding_provider)
            message_id = f"msg_{uuid.uuid4()}"
            collection.add(
                ids=[message_id],
                documents=[message.content],
                metadatas=[{"role": message.type, "timestamp": self._get_timestamp()}],
                # Embeddings are handled automatically by ChromaDB using the function
                # associated with the collection during the .add() call
            )
            logger.debug(f"Added message {message_id} to session {session_id} in Chroma (Provider hint: {embedding_provider})")
        except Exception as e:
            logger.error(f"Error adding message to session {session_id} in Chroma: {e}", exc_info=True)


    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history."""
        try:
            # Getting the collection doesn't need the hint once created.
            collection = self._get_or_create_collection(session_id) # Uses default hint if creation is needed
            results = collection.get(
                include=["metadatas", "documents"],
                # No reliable sorting by timestamp metadata in Chroma `get`
                # We retrieve all and sort in Python. Limit retrieval if performance is an issue.
                # limit=limit * 2 # Fetch a bit more to be safe? Chroma `get` limit applies differently.
            )

            if not results or not results.get("ids"):
                 logger.debug(f"No history found for session '{session_id}' in Chroma collection.")
                 return []

            messages_data = []
            for id_, meta, doc in zip(results["ids"], results["metadatas"], results["documents"]):
                 if meta and 'timestamp' in meta and 'role' in meta and doc is not None:
                     messages_data.append({"id": id_, "metadata": meta, "document": doc})
                 else:
                     logger.warning(f"Skipping message {id_} in session {session_id} due to missing metadata or document: meta={meta}, doc={doc}")


            messages_data.sort(key=lambda x: x["metadata"]["timestamp"], reverse=False)

            history: list[BaseMessage] = []
            # Iterate through sorted data
            for item in messages_data:
                role = item["metadata"].get("role", "unknown")
                content = item["document"]
                if role == "human":
                    history.append(HumanMessage(content=content))
                elif role == "ai":
                    history.append(AIMessage(content=content))
                elif role == "system":
                     history.append(SystemMessage(content=content))

            final_history = history[-limit:] # Apply limit after sorting
            logger.debug(f"Retrieved and reconstructed {len(final_history)} messages for session {session_id} from Chroma.")
            return final_history

        except Exception as e:
            # Check if it's a "collection not found" type error if possible
            # Based on ChromaDB source, it might raise ValueError for missing collection during get
            if "does not exist" in str(e).lower() or isinstance(e, ValueError):
                 logger.warning(f"Collection for session {session_id} not found during get. Returning empty history.")
                 return []
            logger.error(f"Error retrieving history for session {session_id} from Chroma: {e}", exc_info=True)
            return []

    def _get_timestamp(self) -> float:
        return time.time()

# Removed global instance - rely on dependency injection
# session_store = ChromaSessionStore()
