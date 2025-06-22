# src/research_assistant/prompts/__init__.py
from research_assistant.prompts.analyze_prompt import prompt as SUPERVISOR_INTENT_CLASSIFICATION
from research_assistant.prompts.direct_response_prompt import prompt as ASSISTANT_DIRECT_RESPONSE
from research_assistant.prompts.summarize_history_prompt import prompt as ASSISTANT_HISTORY_SUMMARIZATION
from research_assistant.prompts.summarize_prompt import prompt as ASSISTANT_SUMMARIZATION
from research_assistant.prompts.synthesize_response_prompt import (
    prompt_template_simple as ASSISTANT_SYNTHESIS_SIMPLE_SEARCH,
    prompt_template_complex as ASSISTANT_SYNTHESIS_COMPLEX
)

__all__ = [
    "SUPERVISOR_INTENT_CLASSIFICATION",
    "ASSISTANT_DIRECT_RESPONSE",
    "ASSISTANT_HISTORY_SUMMARIZATION",
    "ASSISTANT_SYNTHESIS_SIMPLE_SEARCH",
    "ASSISTANT_SYNTHESIS_COMPLEX",
    "ASSISTANT_SUMMARIZATION"
]