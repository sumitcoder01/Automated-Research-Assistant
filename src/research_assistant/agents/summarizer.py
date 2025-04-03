import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage # For type hinting messages

logger = logging.getLogger(__name__)

# Optional: Define an output schema if you want structured output,
# but for simple summarization, StrOutputParser is often enough.
# class SummaryOutput(BaseModel):
#     summary: str = Field(description="The concise summary of the provided text.")

def format_content_for_summary(state: GraphState) -> str | None:
    """
    Extracts and formats content from the state suitable for summarization.
    Prioritizes search results, then falls back to message history.
    """
    search_results = state.get("search_results")
    messages = state.get("messages", [])
    content_to_summarize = ""

    if search_results and isinstance(search_results, list):
        # Filter out potential error messages or non-dict items if the search failed partially
        valid_results = [res for res in search_results if isinstance(res, dict) and 'error' not in res]
        if valid_results:
            logger.info(f"Summarizer Node: Found {len(valid_results)} valid search results to summarize.")
            formatted_results = "\n---\n".join([
                f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
                for res in valid_results
            ])
            content_to_summarize = f"Based on the following search results:\n{formatted_results}"
        else:
             logger.warning("Summarizer Node: Search results found, but they seem empty or contain errors.")
             # Fall through to potentially use message history

    # Fallback: Use message history if no valid search results or if preferred
    # Avoid summarizing if only the initial user query exists.
    if not content_to_summarize and messages and len(messages) > 1:
        logger.info("Summarizer Node: No valid search results to summarize, using message history.")
        # Format recent messages (e.g., last 4, excluding the current summarization request if possible)
        # This logic might need adjustment based on how messages are added to state
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-4:] if isinstance(msg, BaseMessage)])
        content_to_summarize = f"Based on the recent conversation history:\n{history_str}"

    if not content_to_summarize:
        logger.warning("Summarizer Node: No suitable content found to summarize.")
        return None

    # Optional: Add the original query for context
    query = state.get('query', 'the user query')
    return f"Original Query Context: {query}\n\nContent to Summarize:\n{content_to_summarize}"


def summarize_node(state: GraphState) -> dict:
    """
    Summarizes content found in the state (search results or messages) using an LLM.
    Updates the 'summary' field in the state.
    """
    logger.info("--- Executing Summarizer Node ---")
    summary_update = {"summary": None, "error": None} # Initialize update dict

    # 1. Extract and Format Content
    formatted_content = format_content_for_summary(state)

    if not formatted_content:
        summary_update["summary"] = "No specific content was available to summarize at this step."
        logger.warning("Summarizer Node: Aborting summary due to lack of content.")
        return summary_update # Return immediately

    # 2. Get LLM Instance
    provider = state.get("llm_provider", "openai") # Use default if not specified
    model = state.get("llm_model")
    try:
        llm = get_llm(provider=provider, model=model)
    except ValueError as e:
        logger.error(f"Summarizer Node: Failed to get LLM instance - {e}")
        summary_update["summary"] = f"Error: Could not initialize LLM for summarization."
        summary_update["error"] = str(e)
        return summary_update
    except Exception as e: # Catch other potential init errors
        logger.error(f"Summarizer Node: Unexpected error getting LLM - {e}", exc_info=True)
        summary_update["summary"] = f"Error: Unexpected issue initializing LLM."
        summary_update["error"] = str(e)
        return summary_update

    # 3. Create Prompt
    # Consider moving template string to a dedicated file in research_assistant/prompts/
    prompt_template_str = """You are an expert research assistant. Your goal is to create a concise and informative summary of the provided text. Focus on the key findings, arguments, or pieces of information relevant to the original user query.

Original User Query Context:
{query}

Content to Summarize:
{context}

Provide a clear and coherent summary below:
Summary:"""

    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    # Using basic String Output Parser
    output_parser = StrOutputParser()

    # 4. Create and Invoke Chain
    chain = prompt | llm | output_parser

    try:
        logger.info(f"Summarizer Node: Invoking LLM ({provider}/{model or 'default'}) for summarization.")
        query_context = state.get('query', 'Not specified')
        # Use invoke for synchronous execution within the node
        summary_result = chain.invoke({
            "query": query_context,
            "context": formatted_content
        })

        if isinstance(summary_result, str) and summary_result:
            logger.info(f"Summarizer Node: Successfully generated summary (length {len(summary_result)}).")
            summary_update["summary"] = summary_result.strip()
        else:
             logger.warning(f"Summarizer Node: LLM produced empty or invalid summary output: {summary_result}")
             summary_update["summary"] = "Summarization process completed, but the result was empty."
             summary_update["error"] = "Empty result from LLM"


    except Exception as e:
        logger.error(f"Summarizer Node: LLM invocation failed - {e}", exc_info=True) # Log traceback
        summary_update["summary"] = f"Error: Failed to generate summary."
        summary_update["error"] = f"LLM Error: {str(e)}"

    # 5. Return the updated part of the state
    logger.info("--- Summarizer Node Finished ---")
    return summary_update #