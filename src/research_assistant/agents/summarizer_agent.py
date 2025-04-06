import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage 

logger = logging.getLogger(__name__)

def format_content_for_summary(state: GraphState) -> str | None:
    """
    Extracts and formats content from the state suitable for summarization.
    Prioritizes search results if available, otherwise uses message history,
    especially if the query asks for it.
    """
    query = state.get("query", "").lower() # Get query for intent check
    search_results = state.get("search_results")
    messages = state.get("messages", []) # History + current query message
    content_to_summarize = ""

    # Check if the user explicitly asked for a conversation summary
    is_history_summary_request = "conversation" in query or "history" in query or "previous" in query or "earlier" in query

    # Prioritize search results UNLESS it's explicitly a history summary request
    if search_results and isinstance(search_results, list) and not is_history_summary_request:
        valid_results = [res for res in search_results if isinstance(res, dict) and 'error' not in res]
        if valid_results:
            logger.info(f"Summarizer Node: Found {len(valid_results)} valid search results to summarize.")
            formatted_results = "\n---\n".join([
                f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
                for res in valid_results
            ])
            content_to_summarize = f"Based on the following search results:\n{formatted_results}"
        else:
             logger.warning("Summarizer Node: Search results found, but they seem empty or contain errors. Checking history.")
             # Fall through

    # Use message history if no valid search results OR if history summary was requested
    # Exclude the *very last* message, which is the user's current request (e.g., "summarize this")
    relevant_history = messages[:-1]
    if not content_to_summarize and relevant_history:
        logger.info("Summarizer Node: Using message history for summarization.")
        # Format the relevant history messages
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in relevant_history if isinstance(msg, BaseMessage)])
        content_to_summarize = f"Based on the conversation history:\n{history_str}"

    if not content_to_summarize:
        logger.warning("Summarizer Node: No suitable content found to summarize (neither search nor history).")
        return None

    # The original query is already part of the history usually, but adding context can help
    original_query_context = state.get('query', 'the user query')
    return f"Original Query Context for this Summary Request: {original_query_context}\n\nContent to Summarize:\n{content_to_summarize}"


def summarize_node(state: GraphState) -> dict:
    """
    Summarizes content found in the state (search results or message history) using an LLM.
    Updates the 'summary' field in the state.
    """
    logger.info("--- Executing Summarizer Node ---")
    summary_update = {"summary": None, "error": None}

    # 1. Extract and Format Content (uses the updated logic above)
    formatted_content = format_content_for_summary(state)

    # 3. Create Prompt (Ensure it's general enough for history or search results)
    prompt_template_str = """You are an expert research assistant. Your goal is to create a concise and informative summary of the provided text, which could be search results OR a conversation history. Focus on the key findings, topics discussed, arguments, or pieces of information relevant to the original user query context if provided.

Original User Query Context for this Summary Request:
{query}

Content to Summarize:
{context}

Provide a clear and coherent summary below:
Summary:"""

    if not formatted_content:
        summary_update["summary"] = "No specific content was available to summarize at this step."
        logger.warning("Summarizer Node: Aborting summary due to lack of content.")
        return summary_update

    provider = state.get("llm_provider", "deepseek") # Or your preferred default
    model = state.get("llm_model")
    try:
        llm = get_llm(provider=provider, model=model)
    except ValueError as e:
        logger.error(f"Summarizer Node: Failed to get LLM instance - {e}")
        return {"summary": f"Error: Could not initialize LLM.", "error": str(e)}
    except Exception as e:
        logger.error(f"Summarizer Node: Unexpected error getting LLM - {e}", exc_info=True)
        return {"summary": f"Error: Unexpected issue initializing LLM.", "error": str(e)}

    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        logger.info(f"Summarizer Node: Invoking LLM ({provider}/{model or 'default'}) for summarization.")
        query_context = state.get('query', 'Not specified')
        summary_result = chain.invoke({
            "query": query_context, # The current query (e.g., "summarize")
            "context": formatted_content # The actual content (history or search)
        })

        if isinstance(summary_result, str) and summary_result:
            logger.info(f"Summarizer Node: Successfully generated summary.")
            summary_update["summary"] = summary_result.strip()
        else:
             logger.warning(f"Summarizer Node: LLM produced empty summary output.")
             summary_update["summary"] = "Summarization process completed, but the result was empty."
             summary_update["error"] = "Empty result from LLM"

    except Exception as e:
        logger.error(f"Summarizer Node: LLM invocation failed - {e}", exc_info=True)
        summary_update["summary"] = f"Error: Failed to generate summary."
        summary_update["error"] = f"LLM Error: {str(e)}"

    logger.info("--- Summarizer Node Finished ---")
    return summary_update