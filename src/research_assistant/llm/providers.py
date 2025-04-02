from typing import Optional
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatDeepseek

from ..config import settings, LLMProvider

class LLMProviderManager:
    @staticmethod
    def get_llm(provider: Optional[LLMProvider] = None, model: Optional[str] = None):
        """
        Get an LLM instance based on the specified provider and model.
        Falls back to default settings if not specified.
        """
        provider = provider or settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_MODEL
        
        if provider == LLMProvider.OPENAI:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            return ChatOpenAI(
                model=model,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.7
            )
            
        elif provider == LLMProvider.ANTHROPIC:
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found")
            return ChatAnthropic(
                model=model,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.7
            )
            
        elif provider == LLMProvider.GEMINI:
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API key not found")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.7
            )
            
        elif provider == LLMProvider.DEEPSEEK:
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API key not found")
            return ChatDeepseek(
                model=model,
                deepseek_api_key=settings.DEEPSEEK_API_KEY,
                temperature=0.7
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 