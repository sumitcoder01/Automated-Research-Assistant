import logging
from research_assistant.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

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
