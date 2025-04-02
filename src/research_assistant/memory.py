import chromadb
from chromadb.utils import embedding_functions
from research_assistant.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import logging
import uuid

logger = logging.getLogger(__name__)

class ChromaSessionStore:
    def __init__(self, path: str = settings.chroma_path):
        self.client = chromadb.PersistentClient(path=path)
        # Using a default embedding function, consider SentenceTransformers or OpenAIEmbeddings
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        # Simple in-memory cache for collections to avoid repeated checks
        self._known_collections = set()
        logger.info(f"ChromaDB client initialized at path: {path}")

    def _get_or_create_collection(self, session_id: str):
        """Gets or creates a ChromaDB collection for the session."""
        if session_id not in self._known_collections:
            try:
                collection = self.client.get_or_create_collection(
                    name=f"session_{session_id}",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"} # Example metadata
                )
                self._known_collections.add(session_id)
                logger.info(f"Accessed or created collection for session: {session_id}")
                return collection
            except Exception as e:
                logger.error(f"Error getting/creating collection for session {session_id}: {e}")
                raise
        else:
            # Assumes collection exists if in cache (optimization)
            return self.client.get_collection(name=f"session_{session_id}")


    def add_message(self, session_id: str, message: BaseMessage):
        """Adds a message to the session history."""
        try:
            collection = self._get_or_create_collection(session_id)
            # Use message content as document, role and type as metadata
            # Generate a unique ID for each message entry
            message_id = f"msg_{uuid.uuid4()}"
            collection.add(
                ids=[message_id],
                documents=[message.content],
                metadatas=[{"role": message.type, "timestamp": self._get_timestamp()}], # Store role type
            )
            logger.debug(f"Added message to session {session_id}: {message.type[:10]}...")
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            # Decide if we should raise or just log

    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history."""
        try:
            collection = self._get_or_create_collection(session_id) # Ensure collection exists
            results = collection.get(
                include=["metadatas", "documents"],
                # A common workaround is fetching more than needed and sorting in Python,
                # or designing IDs to be sortable (e.g., timestamp-based).
                # For simplicity here, we retrieve and rely on insertion order or sort later if needed.
            )

            if not results or not results.get("ids"):
                 return []

            # Combine metadata and documents, sorting by timestamp if available
            messages_data = sorted(
                [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(results["ids"], results["metadatas"], results["documents"])
                    if meta and 'timestamp' in meta # Ensure timestamp exists
                ],
                key=lambda x: x["metadata"]["timestamp"],
                reverse=False # Oldest first
            )


            history: list[BaseMessage] = []
            for item in messages_data[-(limit*2):]: # Fetch more to ensure we get pairs if needed, then limit
                role = item["metadata"].get("role", "unknown")
                content = item["document"]
                if role == "human":
                    history.append(HumanMessage(content=content))
                elif role == "ai":
                    history.append(AIMessage(content=content))
                elif role == "system":
                     history.append(SystemMessage(content=content))
                # Add other roles if needed

            logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
            # Return only the last 'limit' messages (can be adjusted based on token limits)
            return history[-limit:]

        except Exception as e:
            # Handle case where collection might not exist yet for a GET request
            if "does not exist" in str(e):
                 logger.warning(f"Collection for session {session_id} not found. Returning empty history.")
                 return []
            logger.error(f"Error retrieving history for session {session_id}: {e}")
            return [] # Return empty list on error


    def _get_timestamp(self) -> float:
        import time
        return time.time()

# Global instance (or use FastAPI dependency injection)
session_store = ChromaSessionStore()