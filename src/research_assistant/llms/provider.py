from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic

from research_assistant.config import settings
from langchain_core.language_models import BaseChatModel

def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """
    Factory function to get an initialized LangChain chat model.
    """
    provider = provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found in settings.")
        # Use default model if none specified, or pass the specific one
        model_name = provider and model or "gpt-4o"
        return ChatOpenAI(api_key=settings.openai_api_key, model=model_name, temperature=0.7) # Example temp

    elif provider == "google" or provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("Google API key not found in settings.")
        model_name = provider and model or "gemini-2.0-flash" # Default Gemini model
        return ChatGoogleGenerativeAI(google_api_key=settings.google_api_key, model=model_name, temperature=0.7)

    elif provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("Deepseek API key not found in settings.")
        model_name = provider and model or "deepseek-chat" # Check actual model names
        return ChatDeepSeek(api_key=settings.deepseek_api_key, model_name=model_name, temperature=0.7)

    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not found in settings.")
        model_name = provider and model or "claude-3-sonnet-20240229" # Example Claude model
        return ChatAnthropic(api_key=settings.anthropic_api_key, model_name=model_name, temperature=0.7)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Example: Get the default LLM
# default_llm = get_llm()