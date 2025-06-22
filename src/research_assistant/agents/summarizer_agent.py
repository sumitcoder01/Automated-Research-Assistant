import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from research_assistant.prompts import ASSISTANT_SUMMARIZATION

logger = logging.getLogger(__name__)

def format_content_for_summary(state: GraphState) -> tuple[str | None, str | None]:
    """
    Extracts and formats content based on the summary request type.
    Prioritizes message history if summary_request_type is 'history'.

    Returns:
        A tuple: (formatted_content, source_type)
    """
    # Check the explicit request type set by the analysis node
    summary_request_type = state.get("summary_request_type") # 'history' or 'content'
    search_results = state.get("search_results")
    messages = state.get("messages", [])
    content_to_summarize = ""
    source_type = None

    # --- Explicitly use history if requested ---
    if summary_request_type == "history":
        logger.info("Summarizer: Explicit request to summarize history detected.")
        # Exclude the *very last* message (the summary request itself)
        relevant_history = messages[:-1]
        if relevant_history:
            source_type = "message_history"
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in relevant_history if isinstance(msg, BaseMessage)])
            # Check for minimal history
            if not history_str.strip():
                 logger.warning("Summarizer: History summary requested, but formatted history is empty.")
                 return None, None # Avoid summarizing just the request
            content_to_summarize = f"The following is the conversation history to be summarized:\n{history_str}"
        else:
            # Handle the edge case where the *only* message is the summary request
            logger.warning("Summarizer: History summary requested, but no prior messages found besides the request itself.")
            return None, None # No actual history to summarize

    # --- Otherwise (request type is 'content' or None), prioritize search results ---
    # This covers cases like "summarize these results" or when routing decides summary after search
    elif search_results and isinstance(search_results, list):
        valid_results = [res for res in search_results if isinstance(res, dict) and 'error' not in res and res.get('snippet')]
        if valid_results:
            logger.info(f"Summarizer: Found {len(valid_results)} valid search results to summarize.")
            source_type = "search_results"
            formatted_results = "\n\n---\n\n".join([
                f"Source {idx+1} (Title: {res.get('title', 'N/A')} | URL: {res.get('link', 'N/A')}):\n{res.get('snippet', 'N/A')}"
                for idx, res in enumerate(valid_results)
            ])
            content_to_summarize = f"The following information was retrieved from web search results:\n{formatted_results}"
        else:
             logger.warning("Summarizer: Content summary likely intended, but search results are empty/invalid.")
             # Don't fall back to history here unless explicitly requested above

    # Fallback if no history summary was requested and no valid search results exist
    if not content_to_summarize:
        logger.warning("Summarizer: No suitable content found to summarize (neither history requested nor valid search results found).")
        return None, None

    return content_to_summarize, source_type


def summarize_node(state: GraphState) -> dict:
    """
    Condenses content based on the determined source (history or search results).
    Updates the 'summary' field in the state.
    """
    logger.info("--- Executing Summarizer Agent Node ---")
    summary_update = {"summary": None, "error": None}

    # 1. Extract and Format Content (Now prioritizes history if requested)
    formatted_content, source_type = format_content_for_summary(state)

    if not formatted_content or not source_type:
        # Handle the edge case where history summary was requested but history was empty
        if state.get("summary_request_type") == "history":
             summary_update["summary"] = "You asked me to summarize the conversation, but there are no previous messages in our chat history to summarize yet!"
             logger.warning("Summarizer Node: Responding that history is empty.")
        else:
             summary_update["summary"] = "No specific content was available to summarize at this step."
             logger.warning("Summarizer Node: Aborting summary due to lack of content/source.")
        return summary_update
    
    provider = state.get("llm_provider")
    model = state.get("llm_model")

    try:
        llm = get_llm(provider=provider, model=model)
    except Exception as e:
        logger.error(f"Summarizer Node: Failed to get LLM instance - {e}", exc_info=True)
        return {"summary": f"Error: Could not initialize LLM.", "error": str(e)}

    # 2. Define Prompt (remains largely the same, adaptable to source)
    prompt = ASSISTANT_SUMMARIZATION
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        logger.info(f"Summarizer Node: Invoking LLM ({provider}/{model or 'default'}) to summarize {source_type}.")
        original_query = state.get('query', 'User requested a summary')

        summary_result = chain.invoke({
            "query": original_query,
            "source_type": source_type,
            "context": formatted_content
        })

        if isinstance(summary_result, str) and summary_result:
            final_summary = summary_result.strip()
            if final_summary.lower().startswith("summary:"):
                final_summary = final_summary[len("summary:"):].strip()

            # Add a specific intro if summarizing history
            if source_type == "message_history":
                 final_summary = f"Okay, here's a summary of our conversation so far:\n\n{final_summary}"

            logger.info("Summarizer Node: Successfully generated summary.")
            summary_update["summary"] = final_summary
        else:
             logger.warning("Summarizer Node: LLM produced empty or non-string summary output.")
             summary_update["summary"] = "Summarization process completed, but the result was empty or invalid."
             summary_update["error"] = "Empty or invalid result from LLM"

    except Exception as e:
        logger.error(f"Summarizer Node: LLM invocation failed - {e}", exc_info=True)
        summary_update["summary"] = "Error: Failed to generate summary due to an LLM error."
        summary_update["error"] = f"LLM Error: {str(e)}"

    logger.info("--- Summarizer Node Finished ---")
    # The summary is intended as the final response in this case
    # We need the graph to route from summarizer to END for history summaries.
    if state.get("summary_request_type") == "history":
         summary_update["final_response"] = summary_update["summary"]

    return summary_update
