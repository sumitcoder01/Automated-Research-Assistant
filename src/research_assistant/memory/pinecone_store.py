import logging
import uuid
import time
from typing import List, Optional
from pinecone import Pinecone, Index, PodSpec
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from research_assistant.config import settings

logger = logging.getLogger(__name__)

# --- Embedding Function Helper (Keep as is) ---
_embedding_cache = {}
def get_embedding_function(provider: str):
    """Initializes and returns a Langchain embedding function based on the provider."""
    provider = provider.lower() if provider else "google" # Default
    cache_key = provider

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    logger.info(f"Initializing embedding function for provider: {provider}")
    embedding_func = None
    if provider == "openai":
        if not settings.openai_api_key:
            logger.error("OpenAI API Key not found in settings for embeddings.")
            raise ValueError("OpenAI API Key missing for embeddings")
        embedding_func = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
    elif provider == "google" or provider == "gemini": # Default case
        if not settings.google_api_key:
            logger.error("Google API Key not found in settings for embeddings.")
            raise ValueError("Google API Key missing for embeddings")
        embedding_func = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key
        )
    else:
         raise ValueError(f"Unsupported embedding provider: {provider}")

    if embedding_func:
        _embedding_cache[cache_key] = embedding_func
    return embedding_func


# --- Pinecone Session Store Implementation ---
# class PineconeSessionStore(BaseSessionStore): # Inherit if Base class exists
class PineconeSessionStore:
    def __init__(self):
        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API Key not configured in settings.")

        try:
            logger.info(f"Initializing Pinecone client...")
            self.pinecone = Pinecone(
                api_key=settings.pinecone_api_key
            )

            index_name = settings.pinecone_index_name
            logger.info(f"Connecting to Pinecone index: '{index_name}'")

            # --- CORRECTED INDEX CHECK ---
            try:
                # List all indexes
                indexes_response = self.pinecone.list_indexes()
                # Extract names from the response object
                existing_index_names = [index_info.name for index_info in indexes_response.indexes]
                logger.debug(f"Found existing Pinecone indexes: {existing_index_names}")

                # Check if the desired index name is in the extracted list
                if index_name not in existing_index_names:
                    logger.error(f"Pinecone index '{index_name}' does not exist in list {existing_index_names}. Please create it first.")
                    raise ValueError(f"Pinecone index '{index_name}' not found.")

            except Exception as list_e: # Catch potential errors during list_indexes
                 logger.error(f"Failed to list Pinecone indexes: {list_e}", exc_info=True)
                 raise ConnectionError(f"Failed to verify Pinecone index existence: {list_e}")
            # --- END OF CORRECTION ---

            self.index: Index = self.pinecone.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'.")

            try:
                 stats = self.index.describe_index_stats()
                 logger.info(f"Index Stats: {stats}")
            except Exception as desc_e:
                 logger.warning(f"Could not describe index stats: {desc_e}")

        except Exception as e:
            logger.error(f"Error during Pinecone initialization ({type(e).__name__}): {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize or connect to Pinecone: {e}")

    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history in Pinecone."""
        try:
            logger.debug(f"Adding message to session '{session_id}' in Pinecone (Provider: {embedding_provider})...")
            embed_function = get_embedding_function(embedding_provider)
            logger.debug(f"Generating embedding for message type: {message.type}")
            vector = embed_function.embed_documents([message.content])[0]
            logger.debug(f"Generated embedding of dimension: {len(vector)}")

            message_id = f"msg_{uuid.uuid4()}"
            timestamp = time.time()
            metadata = {
                "session_id": session_id,
                "role": message.type,
                "timestamp": timestamp,
                "content": message.content
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
                namespace=session_id
            )
            logger.debug(f"Pinecone upsert response: {upsert_response}")
            logger.info(f"Successfully added message '{message_id}' to session '{session_id}'.")

        except ValueError as ve: # Still catch specific config errors if possible
             logger.error(f"Configuration error adding message to session {session_id}: {ve}")
        # --- CATCH GENERAL EXCEPTION for add operation ---
        except Exception as e:
            # Log the actual exception type for debugging
            logger.error(f"Unexpected error adding message ({type(e).__name__}) to session {session_id}: {e}", exc_info=True)
            # Decide if you want to raise this or just log it. Logging allows the request to potentially continue.


    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history from Pinecone."""
        try:
            logger.debug(f"Retrieving history for session '{session_id}' from Pinecone namespace...")
            try:
                 stats = self.index.describe_index_stats()
                 dimension = stats.dimension
                 logger.debug(f"Using index dimension: {dimension}")
            except Exception:
                 logger.warning("Could not get index dimension, assuming 1536 (OpenAI default). Adjust if needed.")
                 dimension = 1536 # Fallback - Ensure this matches your index!

            query_vector = [0.0] * dimension
            MAX_FETCH_LIMIT = 1000
            query_response = self.index.query(
                vector=query_vector,
                top_k=MAX_FETCH_LIMIT,
                namespace=session_id,
                include_metadata=True,
                include_values=False
            )
            logger.debug(f"Pinecone query response received with {len(query_response.get('matches', []))} matches.")

            matches = query_response.get("matches", [])
            if not matches:
                 logger.info(f"No history found for session '{session_id}' in Pinecone.")
                 return []

            messages_data = []
            for match in matches:
                metadata = match.get("metadata", {})
                if "timestamp" in metadata and "role" in metadata and "content" in metadata:
                     messages_data.append(metadata)
                else:
                     logger.warning(f"Skipping message {match.get('id')} due to missing metadata: {metadata}")

            messages_data.sort(key=lambda x: x.get("timestamp", 0))

            history: list[BaseMessage] = []
            for item in messages_data:
                role = item.get("role")
                content = item.get("content")
                if role == "human":
                    history.append(HumanMessage(content=content))
                elif role == "ai":
                    history.append(AIMessage(content=content))
                elif role == "system":
                     history.append(SystemMessage(content=content))

            final_history = history[-limit:]
            logger.info(f"Retrieved and reconstructed {len(final_history)} messages for session '{session_id}'.")
            return final_history

        # --- CATCH GENERAL EXCEPTION for get operation ---
        except Exception as e:
            # Log the actual exception type for debugging
            logger.error(f"Unexpected error retrieving history ({type(e).__name__}) for session {session_id}: {e}", exc_info=True)
            return [] # Return empty list on error