from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # LLM Configuration
    DEFAULT_LLM_PROVIDER: LLMProvider = LLMProvider.DEEPSEEK
    DEFAULT_MODEL: str = "deepseek-chat"
    
    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    
    # FastAPI Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Web Search Configuration
    SEARX_URL: str = "https://searx.be"  # Default to a public instance
    
    # Langsmit Configuration
    LANGSMIT_API_KEY: Optional[str] = None
    LANGSMIT_PROJECT_ID: Optional[str] = None
    
    # Langsmith Configuration
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str
    
    # Application Settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 