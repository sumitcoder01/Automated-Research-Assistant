import os
import tempfile
import shutil
import logging
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from langchain.text_splitter import RecursiveCharacterTextSplitter

from research_assistant.schemas.document import DocumentExtractionResponse
from research_assistant.api.deps import get_session_store
from research_assistant.memory.base import BaseSessionStore
from research_assistant.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@router.post("/upload", response_model=DocumentExtractionResponse)
async def upload_and_process_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...), # Changed variable name to be more descriptive
    store: BaseSessionStore = Depends(get_session_store),
    embedding_provider: str = Form(...)
):
    """
    Receives documents, classifies them, extracts text using the appropriate method,
    chunks the text, saves chunks via the session store, and returns the results.
    """
    processed_files_info = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    embedding_provider = embedding_provider.lower() if embedding_provider else "google"
    logger.info(f"Processing document upload for session '{session_id}' using embedding provider '{embedding_provider}'")

    for file in files:
        # Create a temporary file with the correct extension for processing
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            logger.info(f"Processing file: {file.filename} (temp path: {tmp_file_path})")

            # Step 1: Instantiate the processor
            processor = DocumentProcessor(file_path=tmp_file_path, original_filename=file.filename)
            
            # Step 2: Extract text using the classified method
            full_extracted_text = processor.extract_text()
            
            # Clean text just in case
            cleaned_text = full_extracted_text.encode('utf-8', errors='replace').decode('utf-8')

            # Step 3: Chunking
            text_chunks = text_splitter.split_text(cleaned_text)
            logger.info(f"Extracted {len(cleaned_text)} chars, split into {len(text_chunks)} chunks for '{file.filename}'.")

            # Step 4: Store chunks if any exist
            if text_chunks:
                # Assuming your store has an add_document_chunks method
                # This needs to be implemented in your Chroma/Pinecone stores
                store.add_document_chunks(
                    session_id=session_id,
                    chunks=text_chunks,
                    filename=file.filename,
                    embedding_provider=embedding_provider
                )
                logger.info(f"Successfully processed and stored chunks for '{file.filename}'.")
            else:
                logger.warning(f"No text chunks were generated for '{file.filename}'. It may be empty or contain only images/unreadable content.")

            # Add to list of successfully processed files regardless of chunk count
            processed_files_info.append({"filename": file.filename, "text": cleaned_text})

        except (ValueError, EnvironmentError) as e:
            # Catch specific, user-facing errors from the processor
            logger.error(f"Could not process '{file.filename}': {e}")
            # For now, we continue to the next file but could raise an error to stop the batch
            # Optionally, you could collect these errors and return them in the response.
            # For simplicity, we're skipping failed files.
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing '{file.filename}': {e}", exc_info=True)
            # Optionally skip or raise HTTPException
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            await file.close()

    if not processed_files_info:
        raise HTTPException(status_code=400, detail="No files could be successfully processed. Please check file types and content.")

    # Prepare and return the response
    response_filenames = [info["filename"] for info in processed_files_info]
    response_texts = [info["text"] for info in processed_files_info]

    return DocumentExtractionResponse(
        filenames=response_filenames,
        extracted_texts=response_texts
    )