import uvicorn
from fastapi import FastAPI
import logging.config
import os

# --- Logging Configuration (Basic Example) ---
# For more robust logging, consider a dictionary config or external file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply Langsmith configuration by importing config first
# This ensures environment variables for Langsmith are set early
try:
    from research_assistant.config import settings
    # The print statement about Langsmith status is in config.py
except ImportError as e:
    logger.error(f"Failed to import settings, configuration might be incomplete: {e}")
    # Handle missing config gracefully or exit if critical
    settings = None # Indicate settings are missing

# Import API routers AFTER config is potentially loaded
try:
    from research_assistant.api.endpoints import query, session
except ImportError as e:
     logger.error(f"Failed to import API endpoints: {e}")
     # Decide how to handle this - maybe the app can't start


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Automated Research Assistant API",
    description="API for the multi-agent research assistant using LangGraph and Langsmith.",
    version="0.1.0",
)

# --- API Router ---
app.include_router(session.router, prefix="/api/v1/sessions")
app.include_router(query.router, prefix="/api/v1/query")

# --- Root Endpoint ---
@app.get("/", tags=["Status"])
async def read_root():
    """Basic status check endpoint."""
    return {"status": "Automated Research Assistant API is running!"}

# --- Startup / Shutdown Events (Optional) ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # Initialize any resources if needed (e.g., database connections if not handled by client libraries)
    if not settings:
         logger.critical("Settings could not be loaded. Application might not function correctly.")
    # Ensure ChromaDB path exists? (PersistentClient usually handles this)
    chroma_dir = getattr(settings, 'chroma_path', './chroma_db')
    os.makedirs(chroma_dir, exist_ok=True)
    logger.info(f"Ensured ChromaDB directory exists: {chroma_dir}")
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    # Clean up resources if needed
    logger.info("Application shutdown complete.")


# --- Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        "src.research_assistant.main:app", # Point to the FastAPI app instance
        host="0.0.0.0",
        port=8000,
        reload=True, # Enable auto-reload for development
        log_level="info" # Uvicorn's log level
        )