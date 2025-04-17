# src/research_assistant/memory/pinecone_store.py
import logging
import uuid
import time
from typing import List
from pinecone import Pinecone, Index
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from research_assistant.config import settings
# Import the base class and the shared embedding helper
from research_assistant.memory.base import BaseSessionStore
from research_assistant.utils.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

# --- Pinecone Session Store Implementation ---
class PineconeSessionStore(BaseSessionStore): # Implement the protocol
    def __init__(self):
        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API Key not configured in settings.")
        # Add environment check (important for Pinecone)
        if not settings.pinecone_environment:
            raise ValueError("Pinecone Environment not configured in settings.")

        try:
            logger.info(f"Initializing Pinecone client...")
            self.pinecone = Pinecone(
                api_key=settings.pinecone_api_key,
                # environment=settings.pinecone_environment # Pinecone client >= 3.0 doesn't need env here
            )

            index_name = settings.pinecone_index_name
            logger.info(f"Connecting to Pinecone index: '{index_name}'")

            try:
                indexes_response = self.pinecone.list_indexes()
                existing_index_names = [index_info.name for index_info in indexes_response.indexes]
                logger.debug(f"Found existing Pinecone indexes: {existing_index_names}")

                if index_name not in existing_index_names:
                    logger.error(f"Pinecone index '{index_name}' does not exist. Please create it first.")
                    # Consider if auto-creation is desired/safe or if erroring out is better
                    raise ValueError(f"Pinecone index '{index_name}' not found.")

            except Exception as list_e:
                 logger.error(f"Failed to list Pinecone indexes: {list_e}", exc_info=True)
                 raise ConnectionError(f"Failed to verify Pinecone index existence: {list_e}")

            self.index: Index = self.pinecone.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'.")

            try:
                 stats = self.index.describe_index_stats()
                 logger.info(f"Index Stats: {stats}")
                 # Store dimension for get_history
                 self._index_dimension = stats.dimension
            except Exception as desc_e:
                 logger.warning(f"Could not describe index stats: {desc_e}. Dimension check in get_history might fail.")
                 self._index_dimension = None # Indicate dimension is unknown

        except Exception as e:
            logger.error(f"Error during Pinecone initialization ({type(e).__name__}): {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize or connect to Pinecone: {e}")

    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history in Pinecone."""
        try:
            logger.debug(f"Adding message to session '{session_id}' in Pinecone (Provider: {embedding_provider})...")
            # Use the shared embedding function
            embed_function = get_embedding_function(embedding_provider)
            logger.debug(f"Generating embedding for message type: {message.type}")
            # Ensure content is not None or empty before embedding
            if not message.content:
                logger.warning(f"Attempted to add message with empty content to session {session_id}. Skipping.")
                return
            vector = embed_function.embed_documents([message.content])[0]
            logger.debug(f"Generated embedding of dimension: {len(vector)}")

            message_id = f"msg_{uuid.uuid4()}"
            timestamp = time.time()
            metadata = {
                "session_id": session_id,
                "role": message.type,
                "timestamp": timestamp,
                "content": message.content # Store content for easier retrieval
            }
            logger.debug(f"Prepared metadata: {metadata}")

            logger.debug(f"Upserting vector '{message_id}' to namespace '{session_id}'...")
            upsert_response = self.index.upsert(
                vectors=[
                    {
                        "id": message_id,
                        "values": vector,
                        "metadata": metadata
                    }
                ],
                namespace=session_id # Use session_id as namespace
            )
            logger.debug(f"Pinecone upsert response: {upsert_response}")
            logger.info(f"Successfully added message '{message_id}' to session '{session_id}'.")

        except ValueError as ve:
             logger.error(f"Configuration error adding message to session {session_id}: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error adding message ({type(e).__name__}) to session {session_id}: {e}", exc_info=True)


    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history from Pinecone."""
        try:
            logger.debug(f"Retrieving history for session '{session_id}' from Pinecone namespace...")

            # Use stored dimension if available, otherwise try to fetch it again
            dimension = self._index_dimension
            if dimension is None:
                try:
                    stats = self.index.describe_index_stats()
                    dimension = stats.dimension
                    self._index_dimension = dimension # Cache it now
                    logger.debug(f"Retrieved index dimension dynamically: {dimension}")
                except Exception:
                    logger.warning("Could not get index dimension, assuming 1536 (text-embedding-3-small). Adjust if needed.")
                    dimension = 1536 # Fallback, less reliable

            # Querying with a zero vector to retrieve by metadata sorting is NOT standard in Pinecone.
            # Pinecone is designed for semantic search. A common workaround is to fetch a larger number
            # of recent vectors (if IDs/timestamps allow) or simply fetch by ID if possible.
            # Here, we fetch a high number and sort by timestamp metadata.
            # This assumes timestamps are reliable.
            # Note: Pinecone doesn't guarantee order without specific querying techniques.
            # Fetching a larger number increases latency and cost.
            MAX_FETCH_LIMIT = 1000 # Fetch more candidates to sort later
            logger.debug(f"Querying namespace '{session_id}' with zero vector (fetching {MAX_FETCH_LIMIT} candidates)...")
            query_vector = [0.0] * dimension
            query_response = self.index.query(
                vector=query_vector, # Dummy vector
                top_k=min(MAX_FETCH_LIMIT, 10000), # Pinecone limit might be 10k
                namespace=session_id,
                include_metadata=True,
                include_values=False # Don't need vectors here
            )
            logger.debug(f"Pinecone query response received with {len(query_response.get('matches', []))} matches.")

            matches = query_response.get("matches", [])
            if not matches:
                 logger.info(f"No history found for session '{session_id}' in Pinecone namespace.")
                 return []

            messages_data = []
            for match in matches:
                metadata = match.get("metadata", {})
                # Ensure all necessary fields are present
                if "timestamp" in metadata and "role" in metadata and "content" in metadata:
                     messages_data.append(metadata)
                else:
                     logger.warning(f"Skipping message {match.get('id')} due to missing metadata fields: {metadata}")

            # Sort by timestamp IN metadata
            messages_data.sort(key=lambda x: x.get("timestamp", 0))

            history: list[BaseMessage] = []
            for item in messages_data: # Iterate through sorted data
                role = item.get("role")
                content = item.get("content")
                if role == "human":
                    history.append(HumanMessage(content=content))
                elif role == "ai":
                    history.append(AIMessage(content=content))
                elif role == "system":
                     history.append(SystemMessage(content=content))
                # Add other roles if necessary

            final_history = history[-limit:] # Apply the limit AFTER sorting
            logger.info(f"Retrieved and reconstructed {len(final_history)} messages for session '{session_id}'.")
            return final_history

        except Exception as e:
            logger.error(f"Unexpected error retrieving history ({type(e).__name__}) for session {session_id}: {e}", exc_info=True)
            return [] # Return empty list on error
