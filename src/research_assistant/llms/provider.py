from typing import Optional
from langchain.chat_models import ChatOpenAI, ChatGoogleGenerativeAI
from langchain.schema import BaseMessage

from ..config import settings

def get_llm(provider: Optional[str] = None) -> BaseMessage:
    """
    Get an LLM instance based on the specified provider.
    """
    if provider is None:
        provider = "openai" 
        
    if provider == "openai":
        return ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=settings.GEMINI_API_KEY
        )
    elif provider == "deepseek":
        # Note: Implement DeepSeek integration when available
        raise NotImplementedError("DeepSeek integration not yet available")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 