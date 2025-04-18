# src/research_assistant/memory/chroma_store.py
import chromadb
import logging
import uuid
import time
from typing import List

# Import settings and base class/protocol
from research_assistant.config import settings
from research_assistant.memory.base import BaseSessionStore # Assuming you have base.py

# Import LangChain message types
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

# Import the centralized embedding function getter <--- THIS IS CORRECT
from research_assistant.utils.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

class ChromaSessionStore(BaseSessionStore):
    def __init__(self, path: str = settings.chroma_path):
        try:
            self.client = chromadb.PersistentClient(path=path)
        except Exception as e:
             logger.error(f"Failed to initialize ChromaDB PersistentClient at path '{path}': {e}", exc_info=True)
             # Depending on desired behavior, either raise or handle gracefully
             raise ConnectionError(f"Could not initialize ChromaDB client: {e}")

        self._known_collections = set()
        self._known_doc_collections = set()
        logger.info(f"ChromaDB client initialized at path: {path}")

    # This method correctly uses the imported get_embedding_function
    def _get_or_create_session_collection(self, session_id: str, embedding_provider_hint: str = "google"):
        collection_name = session_id
        # --- Combined get/create logic ---
        try:
            # Try to get first - this is often faster if it exists
            collection = self.client.get_collection(name=collection_name)
            self._known_collections.add(collection_name) # Ensure cache is updated
            logger.debug(f"Retrieved existing session collection '{collection_name}'")
            return collection
        except Exception as get_e: # Broad exception catch as Chroma's specific exceptions can vary
            logger.info(f"Session collection '{collection_name}' not found or error getting ({type(get_e).__name__}), attempting creation...")
            # If get fails, try creating
            try:
                # Get the Langchain embedding function object FROM THE UTILITY
                embedding_func = get_embedding_function(embedding_provider_hint)
                # Pass the object directly to ChromaDB
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func, # Correct usage
                    metadata={"hnsw:space": "cosine"}
                )
                self._known_collections.add(collection_name)
                logger.info(f"Created session collection '{collection_name}' using {embedding_func.__class__.__name__}")
                return collection
            except ValueError as ve: # Catch config errors from get_embedding_function
                logger.error(f"Config error creating session collection {collection_name}: {ve}")
                raise # Re-raise config errors as they likely need user intervention
            except Exception as create_e:
                logger.error(f"Failed to create session collection {collection_name}: {create_e}", exc_info=True)
                raise ConnectionError(f"Failed to create ChromaDB collection {collection_name}: {create_e}") # Raise a standard error

    # This method also correctly uses the imported get_embedding_function
    def _get_or_create_document_collection(self, session_id: str, embedding_provider_hint: str):
        collection_name = session_id
        # --- Combined get/create logic ---
        try:
            collection = self.client.get_collection(name=collection_name)
            self._known_doc_collections.add(collection_name)
            logger.debug(f"Retrieved existing document collection '{collection_name}'")
            return collection
        except Exception as get_e:
            logger.info(f"Document collection '{collection_name}' not found or error getting ({type(get_e).__name__}), attempting creation...")
            try:
                # Get the Langchain embedding function object FROM THE UTILITY
                embedding_func = get_embedding_function(embedding_provider_hint)
                # Pass the object directly to ChromaDB
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func, # Correct usage
                    metadata={"hnsw:space": "cosine"}
                )
                self._known_doc_collections.add(collection_name)
                logger.info(f"Created document collection '{collection_name}' using {embedding_func.__class__.__name__}")
                return collection
            except ValueError as ve:
                logger.error(f"Config error creating document collection {collection_name}: {ve}")
                raise
            except Exception as create_e:
                logger.error(f"Failed to create document collection {collection_name}: {create_e}", exc_info=True)
                raise ConnectionError(f"Failed to create ChromaDB collection {collection_name}: {create_e}")

    # This method correctly uses the collection returned by _get_or_create_session_collection
    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        try:
            if not message.content:
                logger.warning(f"Skipping message with empty content for session {session_id}.")
                return
            cleaned_content = message.content.encode('utf-8', errors='replace').decode('utf-8')
            if cleaned_content != message.content:
                 logger.warning(f"Replaced invalid UTF-8 characters in message for session {session_id}")

            # This will create the collection with the correct embedding func if it doesn't exist
            collection = self._get_or_create_session_collection(session_id, embedding_provider_hint=embedding_provider)
            message_id = f"msg_{uuid.uuid4()}"
            collection.add(
                ids=[message_id],
                documents=[cleaned_content],
                metadatas=[{"role": message.type, "timestamp": self._get_timestamp()}],
            )
            logger.debug(f"Added message {message_id} to session collection '{collection.name}'.")
        except ValueError as ve: # Catch config errors bubbled up from _get_or_create
            logger.error(f"Config error adding message to session {session_id}: {ve}")
            # Raise a standard error, let API layer handle HTTP exception
            raise ValueError(f"Embedding configuration error: {ve}")
        except ConnectionError as ce: # Catch connection/creation errors
             logger.error(f"Storage connection error adding message to session {session_id}: {ce}")
             raise ConnectionError(f"Failed to access storage for session {session_id}: {ce}")
        except Exception as e:
            logger.error(f"Error adding message to session {session_id} in Chroma: {e}", exc_info=True)
            # Raise a standard error
            raise RuntimeError(f"Internal error storing message for session {session_id}.")


    # This method correctly uses the collection returned by _get_or_create_session_collection
    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        try:
            # This will use the default hint if collection needs creation,
            # otherwise it just gets the existing one.
            collection = self._get_or_create_session_collection(session_id)
            results = collection.get(include=["metadatas", "documents"])

            # ... (rest of history processing logic is fine) ...
            if not results or not results.get("ids"):
                logger.debug(f"No history in session collection for '{session_id}'.")
                return []

            messages_data = [
                {"id": id_, "metadata": meta, "document": doc}
                for id_, meta, doc in zip(results["ids"], results["metadatas"], results["documents"])
                if meta and 'timestamp' in meta and 'role' in meta and doc is not None
            ]

            messages_data.sort(key=lambda x: x["metadata"]["timestamp"], reverse=False)

            history: list[BaseMessage] = []
            for item in messages_data:
                role = item["metadata"].get("role", "unknown")
                content = item["document"]
                if role == "human": history.append(HumanMessage(content=content))
                elif role == "ai": history.append(AIMessage(content=content))
                elif role == "system": history.append(SystemMessage(content=content))

            final_history = history[-limit:]
            logger.debug(f"Retrieved {len(final_history)} messages for session {session_id} from Chroma.")
            return final_history

        except (ValueError, ConnectionError) as e: # Catch errors from _get_or_create
            logger.warning(f"Collection access error for session {session_id} retrieving history: {e}. Returning empty history.")
            return []
        except Exception as e:
            # Catch other potential errors during .get() or processing
            logger.error(f"Unexpected error retrieving history for session {session_id} from Chroma: {e}", exc_info=True)
            # Return empty list on error, endpoint can decide on 500 status
            return []

    # This method correctly uses the collection returned by _get_or_create_document_collection
    def add_document_chunks(self, session_id: str, chunks: List[str], filename: str, embedding_provider: str):
        if not chunks:
            logger.warning(f"No chunks provided for file {filename} in session {session_id}. Skipping.")
            return

        try:
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                cleaned_chunk = chunk.encode('utf-8', errors='replace').decode('utf-8')
                if cleaned_chunk != chunk:
                    logger.warning(f"Replaced invalid UTF-8 characters in chunk {i} for file {filename}, session {session_id}")
                cleaned_chunks.append(cleaned_chunk)

            # This will create the collection with the correct embedding func if it doesn't exist
            doc_collection = self._get_or_create_document_collection(session_id, embedding_provider_hint=embedding_provider)

            ids = [f"doc_{session_id}_{filename}_chunk_{uuid.uuid4()}" for _ in cleaned_chunks]
            metadatas = [{"role": "system", "timestamp": self._get_timestamp()} for i in range(len(cleaned_chunks))]

            logger.info(f"Adding {len(cleaned_chunks)} cleaned chunks from {filename} to document collection '{doc_collection.name}'.")
            doc_collection.add(
                ids=ids,
                documents=cleaned_chunks,
                metadatas=metadatas
            )
            logger.debug(f"Successfully added chunks for {filename} to document collection '{doc_collection.name}'.")

        except ValueError as ve: # Catch config errors bubbled up
            logger.error(f"Config error adding document chunks for session {session_id}, file {filename}: {ve}")
            raise ValueError(f"Embedding configuration error: {ve}")
        except ConnectionError as ce: # Catch storage/connection errors
             logger.error(f"Storage connection error adding chunks for session {session_id}: {ce}")
             raise ConnectionError(f"Failed to access storage for session {session_id}: {ce}")
        except Exception as e:
            if isinstance(e, UnicodeEncodeError): # Keep specific check
                logger.error(f"UnicodeEncodeError during document chunk processing for {filename}: {e}", exc_info=True)
                # Raise a specific error type the API could potentially catch
                raise TypeError(f"Invalid characters found in file {filename}. Could not process.")
            logger.error(f"Failed to add document chunks for session {session_id}, file {filename}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to store document chunks for session {session_id}.")


    def _get_timestamp(self) -> float:
        return time.time()