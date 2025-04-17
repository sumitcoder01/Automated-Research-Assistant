# src/research_assistant/api/endpoints/documents.py
import os
import tempfile
import shutil
import logging
from typing import List, Union, Dict, Any
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from research_assistant.schemas.document import DocumentExtractionResponse
from research_assistant.config import settings
# Import the new dependency and the embedding helper utility
from research_assistant.api.deps import get_vector_store_backend
from research_assistant.utils.embeddings import get_embedding_function

# Import backend types for type hinting
import chromadb
from pinecone import Index # Import Pinecone Index

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Constants ---
# No longer using a single global collection name
# DOCUMENT_COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper Functions (Text Extraction - Keep As Is) ---
# ... (extract_text_from_pdf, extract_text_from_docx, extract_text_from_image remain the same) ...
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file, attempting OCR if direct extraction fails."""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if len(text.strip()) < 50:
             logger.warning(f"Minimal text extracted directly from PDF {os.path.basename(file_path)}, attempting OCR.")
             try:
                 images = convert_from_path(file_path)
                 ocr_text = ""
                 for i, img in enumerate(images):
                     logger.debug(f"Running OCR on page {i+1} of {os.path.basename(file_path)}")
                     ocr_text += pytesseract.image_to_string(img) + " "
                 if len(ocr_text.strip()) > len(text.strip()):
                     logger.info(f"OCR provided more text for {os.path.basename(file_path)}.")
                     text = ocr_text
                 else:
                     logger.warning(f"OCR did not yield significantly more text for {os.path.basename(file_path)}.")
             except Exception as ocr_error:
                 logger.error(f"Error during OCR processing for {os.path.basename(file_path)}: {ocr_error}", exc_info=True)
    except Exception as e:
        logger.error(f"Error reading PDF file {os.path.basename(file_path)}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {os.path.basename(file_path)}") from e
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX file {os.path.basename(file_path)}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX file: {os.path.basename(file_path)}") from e
    return text

def extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image file using OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        logger.error(f"Error during OCR for image file {os.path.basename(file_path)}: {e}", exc_info=True)
        if "Tesseract is not installed" in str(e) or "tesseract not found" in str(e).lower():
             logger.critical("Tesseract executable not found. Ensure Tesseract is installed and pytesseract.tesseract_cmd is set correctly if needed.")
             raise HTTPException(status_code=500, detail="Tesseract OCR engine not found or configured correctly.")
        raise HTTPException(status_code=500, detail=f"Failed to process image file: {os.path.basename(file_path)}") from e
    return text


# --- UPDATED Helper Function for Storing Chunks ---

def store_document_chunks(
    backend: Union[chromadb.Client, Index],
    session_id: str,
    chunks: List[str],
    filename: str,
    embedding_provider: str # Added provider hint
):
    """Stores text chunks in the appropriate vector store (Chroma or Pinecone) for a given session."""
    if not chunks:
        logger.warning(f"No chunks to store for {filename} in session {session_id}.")
        return

    # Determine embedding function based on hint
    try:
        embed_function = get_embedding_function(embedding_provider)
        # Simple check for embedding function result dimensions if needed (optional)
        # test_embedding = embed_function.embed_query("test")
        # logger.debug(f"Embedding dimension for provider {embedding_provider}: {len(test_embedding)}")
    except ValueError as e:
        logger.error(f"Failed to get embedding function for provider {embedding_provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding configuration error: {e}")

    # Generate unique IDs for each chunk
    ids = [f"doc_{session_id}_{filename}_chunk_{uuid.uuid4()}" for _ in chunks]
    metadatas = [{"source": filename, "session_id": session_id, "chunk_index": i} for i in range(len(chunks))]

    # --- Chroma Logic ---
    if isinstance(backend, chromadb.Client):
        collection_name = f"docs_{session_id}" # Collection per session for documents
        try:
            logger.debug(f"Getting or creating Chroma collection: {collection_name} for session {session_id}")
            # Use Chroma's utility to wrap the Langchain embedding function
            chroma_embedding_func = chromadb.utils.embedding_functions.LangchainEmbeddingFunction(embed_function)

            collection = backend.get_or_create_collection(
                name=collection_name,
                embedding_function=chroma_embedding_func, # Use the wrapped function
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Adding {len(chunks)} chunks from {filename} to Chroma collection '{collection_name}' for session {session_id}.")
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
                # Embeddings handled by Chroma
            )
            logger.debug(f"Successfully added chunks for {filename} to Chroma session {session_id}.")
        except Exception as e:
            logger.error(f"Failed to add chunks for {filename} to Chroma session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store document chunks in Chroma for session {session_id}.")

    # --- Pinecone Logic ---
    elif isinstance(backend, Index):
        pinecone_index = backend
        namespace = f"docs_{session_id}" # Use a namespace per session for documents
        try:
            logger.debug(f"Generating embeddings for {len(chunks)} chunks using {embedding_provider}...")
            embeddings = embed_function.embed_documents(chunks)
            logger.debug(f"Generated {len(embeddings)} embeddings.")

            vectors_to_upsert = []
            for i, chunk_embedding in enumerate(embeddings):
                vectors_to_upsert.append({
                    "id": ids[i],
                    "values": chunk_embedding,
                    "metadata": metadatas[i]
                })

            # Upsert in batches if necessary (Pinecone has limits)
            batch_size = 100 # Pinecone recommended batch size
            logger.info(f"Upserting {len(vectors_to_upsert)} vectors for {filename} to Pinecone index '{pinecone_index.name}' namespace '{namespace}' for session {session_id}.")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                logger.debug(f"Upserting batch {i//batch_size + 1}... (size: {len(batch)})")
                upsert_response = pinecone_index.upsert(vectors=batch, namespace=namespace)
                logger.debug(f"Pinecone upsert response for batch: {upsert_response}")

            logger.debug(f"Successfully upserted chunks for {filename} to Pinecone session {session_id}.")

        except Exception as e:
            logger.error(f"Failed to add chunks for {filename} to Pinecone session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store document chunks in Pinecone for session {session_id}.")

    else:
        logger.error(f"Unsupported vector store backend type: {type(backend)}")
        raise HTTPException(status_code=500, detail="Internal server error: Invalid vector store configuration.")


# --- UPDATED API Endpoint ---

# Changed route from /extract to /upload to be more conventional
@router.post("/upload/documents", response_model=DocumentExtractionResponse)
async def upload_and_process_documents(
    session_id: str = Form(...), # Added session_id from form data
    files: List[UploadFile] = File(...),
    # Inject the low-level backend client/index
    vector_store_backend: Union[chromadb.Client, Index] = Depends(get_vector_store_backend),
    embedding_provider: str = Form(...)
):
    """
    Receives one or more documents (PDF, DOCX, Images) for a specific session,
    extracts text, chunks it, and stores the chunks in the vector store
    associated with the session_id.
    """
    embedding_provider
    extracted_texts = []
    processed_filenames = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # Determine embedding provider - Use a default or potentially get from config/request
    # For simplicity, using a default 'google'. Make this configurable if needed.
    embedding_provider = embedding_provider.lower() if embedding_provider else "google" 
    logger.info(f"Processing document upload for session_id: {session_id} using embedding provider: {embedding_provider_hint}")


    for file in files:
        tmp_file_path = None # Ensure path is defined for finally block
        try:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_file_path = tmp_file.name
            logger.info(f"Processing file: {file.filename} for session {session_id} (temp: {tmp_file_path})")

            full_extracted_text = ""
            content_type = file.content_type
            logger.debug(f"File {file.filename} content type: {content_type}")

            # Extract text (same logic as before)
            if content_type == "application/pdf" or tmp_file_path.lower().endswith(".pdf"):
                full_extracted_text = extract_text_from_pdf(tmp_file_path)
            elif content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ] or tmp_file_path.lower().endswith(".docx"):
                 full_extracted_text = extract_text_from_docx(tmp_file_path)
            elif content_type.startswith("image/") or any(tmp_file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']):
                 full_extracted_text = extract_text_from_image(tmp_file_path)
            else:
                logger.warning(f"Unsupported file type for {file.filename}: {content_type}. Skipping.")
                raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}: {content_type}")


            # Chunk the extracted text
            text_chunks = text_splitter.split_text(full_extracted_text)
            logger.info(f"Extracted {len(full_extracted_text)} chars, split into {len(text_chunks)} chunks for {file.filename} (session: {session_id}).")

            # Store chunks using the updated function and injected backend
            if text_chunks:
                store_document_chunks(
                    backend=vector_store_backend,
                    session_id=session_id,
                    chunks=text_chunks,
                    filename=file.filename,
                    embedding_provider=embedding_provider
                )
            else:
                logger.warning(f"No text could be extracted from {file.filename} for session {session_id}, skipping storage.")

            # Storing the full extracted text might be memory intensive for large files
            # Consider only storing filename or success status
            extracted_texts.append({"filename": file.filename, "status": "processed"}) # Changed to status
            processed_filenames.append(file.filename)

        except HTTPException as http_exc:
            logger.error(f"HTTP Exception during processing {file.filename} for session {session_id}: {http_exc.detail}")
            # Re-raise to let FastAPI handle it
            raise http_exc
        except Exception as e:
            logger.error(f"An unexpected error occurred processing file {file.filename} for session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred processing {file.filename} for session {session_id}.")
        finally:
            # Cleanup
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                    logger.debug(f"Removed temporary file: {tmp_file_path}")
                except OSError as rm_error:
                    logger.error(f"Error removing temporary file {tmp_file_path}: {rm_error}")
            await file.close()

    # Adjust the response schema if necessary
    # Returning just filenames seems reasonable
    return DocumentExtractionResponse(
        filenames=processed_filenames,
        # extracted_texts=[t['content'] for t in extracted_texts] # Removed extracted text content
        extracted_texts=[] # Return empty list or adjust schema as needed
    )

