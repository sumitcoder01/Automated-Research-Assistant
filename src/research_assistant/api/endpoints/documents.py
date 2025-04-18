# src/research_assistant/api/endpoints/documents.py
import os
import tempfile
import shutil
import logging
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from research_assistant.schemas.document import DocumentExtractionResponse
# Use the SessionStore dependency instead of the low-level one
from research_assistant.api.deps import get_session_store
from research_assistant.memory.base import BaseSessionStore

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper Functions (Text Extraction - Can remain mostly the same) ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file, attempting OCR if direct extraction fails."""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        # OCR Fallback logic
        if len(text.strip()) < 50:
             logger.warning(f"Minimal text from PDF {os.path.basename(file_path)}, attempting OCR.")
             try:
                 images = convert_from_path(file_path)
                 ocr_text = "".join(pytesseract.image_to_string(img) + " " for img in images)
                 if len(ocr_text.strip()) > len(text.strip()):
                     logger.info(f"OCR provided more text for {os.path.basename(file_path)}.")
                     text = ocr_text
             except Exception as ocr_error:
                 logger.error(f"Error during OCR for {os.path.basename(file_path)}: {ocr_error}")
    except Exception as e:
        logger.error(f"Error reading PDF file {os.path.basename(file_path)}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {os.path.basename(file_path)}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX file {os.path.basename(file_path)}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {os.path.basename(file_path)}")

def extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image file using OCR."""
    try:
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        logger.error(f"Error during OCR for image {os.path.basename(file_path)}: {e}", exc_info=True)
        if "Tesseract is not installed" in str(e) or "tesseract not found" in str(e).lower():
             raise HTTPException(status_code=500, detail="Tesseract OCR engine not found.")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {os.path.basename(file_path)}")


# --- REFACTORED API Endpoint ---

@router.post("/upload", response_model=DocumentExtractionResponse)
async def upload_and_process_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    store: BaseSessionStore = Depends(get_session_store),
    embedding_provider: str = Form(...)
):
    """
    Receives documents, extracts text, chunks it, saves chunks via session store,
    and returns the list of processed filenames and their full extracted text.
    """
    processed_files_info = [] # Store dicts like {"filename": ..., "text": ...}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    embedding_provider = embedding_provider.lower() if embedding_provider else "google"
    logger.info(f"Processing document upload for session '{session_id}' using embedding provider '{embedding_provider}'")

    for file in files:
        tmp_file_path = None
        full_extracted_text = "" # Initialize for each file
        try:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_file_path = tmp_file.name
            logger.info(f"Processing file: {file.filename} (temp: {tmp_file_path})")

            content_type = file.content_type

            # Extract text
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
                logger.warning(f"Unsupported file type '{content_type}' for {file.filename}. Skipping.")
                continue # Skip to next file

            # Clean extracted text (optional, but good practice before chunking/storing)
            cleaned_extracted_text = full_extracted_text.encode('utf-8', errors='replace').decode('utf-8')
            if cleaned_extracted_text != full_extracted_text:
                 logger.warning(f"Replaced invalid UTF-8 characters in extracted text for {file.filename}")

            # Chunking
            text_chunks = text_splitter.split_text(cleaned_extracted_text)
            logger.info(f"Extracted {len(cleaned_extracted_text)} chars, split into {len(text_chunks)} chunks for {file.filename}.")

            # Store chunks
            if text_chunks:
                try:
                    store.add_document_chunks(
                        session_id=session_id,
                        chunks=text_chunks, # Pass potentially cleaned chunks to store
                        filename=file.filename,
                        embedding_provider=embedding_provider
                    )
                    # Add to list of successfully processed files *after* storing chunks
                    processed_files_info.append({"filename": file.filename, "text": cleaned_extracted_text})
                    logger.info(f"Successfully processed and stored chunks for {file.filename}.")
                except HTTPException as http_exc:
                    logger.error(f"HTTP error storing chunks for {file.filename}: {http_exc.detail}")
                    # Stop processing if storing fails for one file?
                    # Or collect errors and report them? Let's stop for now.
                    raise http_exc
                except Exception as store_exc:
                    logger.error(f"Unexpected error storing chunks for {file.filename}: {store_exc}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to store {file.filename}.")
            else:
                # If no text chunks were generated, still consider the file processed 
                # if text extraction itself didn't fail, but maybe warn.
                logger.warning(f"No text chunks generated for {file.filename} (was text extracted?). Adding to processed list without storing.")
                processed_files_info.append({"filename": file.filename, "text": cleaned_extracted_text})

        except HTTPException as http_exc:
            logger.error(f"HTTP Exception processing {file.filename}: {http_exc.detail}")
            # Stop processing more files if one fails badly
            raise http_exc
        except Exception as e:
            logger.error(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error processing {file.filename}.")
        finally:
            # Cleanup
            if tmp_file_path and os.path.exists(tmp_file_path):
                try: os.remove(tmp_file_path)
                except OSError as rm_error: logger.error(f"Error removing temp file {tmp_file_path}: {rm_error}")
            await file.close()

    if not processed_files_info:
        logger.warning(f"No files were successfully processed for session '{session_id}'.")
        raise HTTPException(status_code=400, detail="No files could be successfully processed.")

    # Prepare response
    response_filenames = [info["filename"] for info in processed_files_info]
    response_texts = [info["text"] for info in processed_files_info]

    return DocumentExtractionResponse(
        filenames=response_filenames,
        extracted_texts=response_texts # Return the collected texts
    )
