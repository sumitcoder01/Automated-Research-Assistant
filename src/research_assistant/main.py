import uvicorn
from fastapi import FastAPI
import logging.config
import os

# --- Logging Configuration ---
LOG_FILE = "app.log"
LOG_LEVEL = logging.INFO # Set desired level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE), # Log to a file
        logging.StreamHandler()      # Also log to console (optional)
    ]
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Apply Langsmith configuration by importing config first
try:
    from research_assistant.config import settings
except ImportError as e:
    logger.error(f"Failed to import settings, configuration might be incomplete: {e}")
    settings = None

# Import API routers AFTER config is potentially loaded
try:
    from research_assistant.api.endpoints import query, session, documents
except ImportError as e:
     logger.error(f"Failed to import API endpoints: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Automated Research Assistant API",
    description="API for the multi-agent research assistant using LangGraph and Langsmith.",
    version="0.1.0",
)

# --- API Routers ---
app.include_router(session.router, prefix="/api/v1/sessions")
app.include_router(query.router, prefix="/api/v1/query")
app.include_router(documents.router, prefix="/api/v1/documents")

# --- Root Endpoint ---
@app.get("/", tags=["Status"])
async def read_root():
    """Basic status check endpoint."""
    return {"status": "Automated Research Assistant API is running!"}

# --- Startup / Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("-"*20 + " Application Startup " + "-"*20)
    if not settings:
         logger.critical("Settings could not be loaded. Application might not function correctly.")
    # Ensure ChromaDB path exists
    try:
        chroma_dir = getattr(settings, 'chroma_path', './chroma_db')
        os.makedirs(chroma_dir, exist_ok=True)
        logger.info(f"Ensured ChromaDB directory exists: {chroma_dir}")
    except Exception as e:
        logger.error(f"Failed to ensure ChromaDB directory: {e}")
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("-"*20 + " Application Shutdown " + "-"*20)
    # Clean up resources if needed
    logger.info("Application shutdown complete.")


# --- Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run(
        "src.research_assistant.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info" # Uvicorn's log level (controls Uvicorn's own messages)
        # Note: Python's logging level is set separately above
    )
