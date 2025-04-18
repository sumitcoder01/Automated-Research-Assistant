# src/research_assistant/memory/pinecone_store.py
import logging
import uuid
import time
from typing import List
from pinecone import Pinecone, Index
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from fastapi import HTTPException # For raising HTTP exceptions

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
        if not settings.pinecone_environment:
            raise ValueError("Pinecone Environment not configured in settings.")

        try:
            logger.info(f"Initializing Pinecone client...")
            self.pinecone = Pinecone(
                api_key=settings.pinecone_api_key,
            )
            index_name = settings.pinecone_index_name
            logger.info(f"Connecting to Pinecone index: '{index_name}'")
            try:
                indexes_response = self.pinecone.list_indexes()
                existing_index_names = [index_info.name for index_info in indexes_response.indexes]
                if index_name not in existing_index_names:
                    logger.error(f"Pinecone index '{index_name}' does not exist. Create it first.")
                    raise ValueError(f"Pinecone index '{index_name}' not found.")
            except Exception as list_e:
                 logger.error(f"Failed to list Pinecone indexes: {list_e}", exc_info=True)
                 raise ConnectionError(f"Failed to verify Pinecone index existence: {list_e}")

            self.index: Index = self.pinecone.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'.")
            try:
                 stats = self.index.describe_index_stats()
                 logger.info(f"Index Stats: {stats}")
                 self._index_dimension = stats.dimension
            except Exception as desc_e:
                 logger.warning(f"Could not describe index stats: {desc_e}. Using fallback dimension.")
                 self._index_dimension = None # Reset on failure

        except Exception as e:
            logger.error(f"Error during Pinecone initialization ({type(e).__name__}): {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize or connect to Pinecone: {e}")

    def _get_dimension(self) -> int:
        """Gets the index dimension, fetching if necessary."""
        if self._index_dimension is None:
            try:
                stats = self.index.describe_index_stats()
                self._index_dimension = stats.dimension
                logger.debug(f"Fetched index dimension: {self._index_dimension}")
            except Exception as e:
                logger.error(f"Failed to fetch Pinecone index dimension: {e}. Falling back to 1536.")
                # Use a common default (e.g., for text-embedding-3-small), adjust if using others
                return 1536
        return self._index_dimension

    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history namespace in Pinecone."""
        namespace = session_id 
        try:
            if not message.content:
                logger.warning(f"Skipping message with empty content for session {session_id}.")
                return

            # Clean message content before embedding
            cleaned_content = message.content.encode('utf-8', errors='replace').decode('utf-8')
            if cleaned_content != message.content:
                logger.warning(f"Replaced invalid UTF-8 characters in message for session {session_id}")

            logger.debug(f"Adding message to namespace '{namespace}' (Provider: {embedding_provider})...")
            embed_function = get_embedding_function(embedding_provider)
            vector = embed_function.embed_documents([cleaned_content])[0] # Embed cleaned content
            message_id = f"msg_{uuid.uuid4()}"
            timestamp = time.time()
            metadata = {
                "session_id": session_id,
                "role": message.type,
                "timestamp": timestamp,
                "content": cleaned_content # Store cleaned content
            }
            logger.debug(f"Upserting message vector '{message_id}' to namespace '{namespace}'.")
            upsert_response = self.index.upsert(
                vectors=[{"id": message_id, "values": vector, "metadata": metadata}],
                namespace=namespace
            )
            logger.debug(f"Pinecone upsert response: {upsert_response}")
            logger.info(f"Successfully added message '{message_id}' to namespace '{namespace}'.")
        except ValueError as ve:
             logger.error(f"Config error adding message to session {session_id}, ns {namespace}: {ve}")
             # Propagate error for API endpoint handling
             raise HTTPException(status_code=500, detail=f"Embedding configuration error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error adding message ({type(e).__name__}) to session {session_id}, ns {namespace}: {e}", exc_info=True)
            # Propagate error for API endpoint handling
            raise HTTPException(status_code=500, detail=f"Internal error storing message for session {session_id}.")


    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves history from the session message namespace in Pinecone."""
        namespace = session_id # Use the message namespace
        try:
            logger.debug(f"Retrieving history from Pinecone namespace '{namespace}'...")
            dimension = self._get_dimension()
            query_vector = [0.0] * dimension
            MAX_FETCH_LIMIT = 1000
            query_response = self.index.query(
                vector=query_vector,
                top_k=min(MAX_FETCH_LIMIT, 10000),
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            matches = query_response.get("matches", [])
            if not matches:
                 logger.info(f"No history found in namespace '{namespace}'.")
                 return []

            messages_data = [
                match.get("metadata", {})
                for match in matches
                if "timestamp" in match.get("metadata", {}) and "role" in match.get("metadata", {}) and "content" in match.get("metadata", {})
            ]
            messages_data.sort(key=lambda x: x.get("timestamp", 0))

            history: list[BaseMessage] = []
            for item in messages_data:
                role = item.get("role")
                content = item.get("content")
                if role == "human": history.append(HumanMessage(content=content))
                elif role == "ai": history.append(AIMessage(content=content))
                elif role == "system": history.append(SystemMessage(content=content))

            final_history = history[-limit:]
            logger.info(f"Retrieved {len(final_history)} messages for session '{session_id}' from ns '{namespace}'.")
            return final_history
        except Exception as e:
            logger.error(f"Error retrieving history ({type(e).__name__}) for ns '{namespace}': {e}", exc_info=True)
            return []

    def add_document_chunks(self, session_id: str, chunks: List[str], filename: str, embedding_provider: str):
        """Adds document chunks to a dedicated document namespace for the session."""
        if not chunks:
            logger.warning(f"No chunks provided for file {filename} in session {session_id}. Skipping storage.")
            return

        namespace = session_id # Dedicated namespace for documents
        logger.info(f"Adding {len(chunks)} chunks from {filename} to Pinecone namespace '{namespace}' using {embedding_provider}.")

        try:
            # --- Clean chunks before embedding --- 
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                cleaned_chunk = chunk.encode('utf-8', errors='replace').decode('utf-8')
                if cleaned_chunk != chunk:
                    logger.warning(f"Replaced invalid UTF-8 characters in chunk {i} for file {filename}, session {session_id}")
                cleaned_chunks.append(cleaned_chunk)
            # --------------------------------------

            embed_function = get_embedding_function(embedding_provider)
            logger.debug(f"Generating embeddings for {len(cleaned_chunks)} cleaned chunks...")
            # Embed the cleaned chunks
            embeddings = embed_function.embed_documents(cleaned_chunks)
            logger.debug(f"Generated {len(embeddings)} embeddings.")

            vectors_to_upsert = []
            current_time = time.time()
            # Use cleaned_chunks when creating metadata if you decide to store content
            for i, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings)):
                chunk_id = f"doc_{session_id}_{filename}_chunk_{uuid.uuid4()}"
                metadata = {
                    "session_id": session_id,
                    "role":"system",
                    "timestamp": current_time,
                    "content": chunk
                }
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
                

            # Upsert in batches
            batch_size = 100 # Pinecone recommended batch size
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                logger.debug(f"Upserting chunk batch {i//batch_size + 1}... (size: {len(batch)}) to ns '{namespace}'")
                upsert_response = self.index.upsert(vectors=batch, namespace=namespace)
                logger.debug(f"Pinecone chunk upsert response for batch: {upsert_response}")

            logger.info(f"Successfully upserted {len(vectors_to_upsert)} chunks for {filename} to Pinecone namespace '{namespace}'.")

        except ValueError as ve:
            logger.error(f"Configuration error adding document chunks for session {session_id}, file {filename}, ns {namespace}: {ve}")
            raise HTTPException(status_code=500, detail=f"Embedding configuration error: {ve}")
        except Exception as e:
            # Catch potential Unicode errors during embedding/processing specifically
            if isinstance(e, UnicodeEncodeError):
                logger.error(f"UnicodeEncodeError during document chunk processing for {filename}, ns {namespace}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Invalid characters in file {filename}. Could not process.")
            logger.error(f"Failed to add document chunks for session {session_id}, file {filename}, ns {namespace}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store document chunks in Pinecone for session {session_id}.")
