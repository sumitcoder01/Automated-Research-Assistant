import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Langsmith
    langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    langchain_endpoint: str | None = os.getenv("LANGCHAIN_ENDPOINT")
    langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str | None = os.getenv("LANGCHAIN_PROJECT", "Automated Research Assistant")

    # LLM API Keys (Load keys securely)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    # App Specific
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")

    # Search API Key
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")


    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

# Create a single settings instance for the application
settings = Settings()

# Configure Langsmith client based on settings
# This ensures Langsmith is ready when modules are imported
if settings.langchain_api_key:
    print("Langsmith tracing enabled.")

else:
    print("Langsmith API key not found. Tracing disabled.")