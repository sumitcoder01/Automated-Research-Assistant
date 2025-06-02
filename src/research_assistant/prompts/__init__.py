# src/research_assistant/prompts/__init__.py
from research_assistant.prompts.analyze_prompt import prompt as Analyze_Prompt
from research_assistant.prompts.direct_response_prompt import prompt as Direct_Response_Prompt
from research_assistant.prompts.summarize_history_prompt import prompt as Summarize_History_Prompt
from research_assistant.prompts.summarize_prompt import prompt as Summarize_Prompt
from research_assistant.prompts.synthesize_response_prompt import (
    prompt_template_simple as Prompt_Template_Simple
    prompt_template_complex as Prompt_Template_Complex
)

__all__ = [
    "Analyze_Prompt",
    "Direct_Response_Prompt",
    "Summarize_History_Prompt",
    "Prompt_Template_Simple",
    "Prompt_Template_Complex",
    "Summarize_Prompt"
]