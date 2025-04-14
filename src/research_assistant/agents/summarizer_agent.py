import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

def format_content_for_summary(state: GraphState) -> tuple[str | None, str | None]:
    """
    Extracts and formats content from the state suitable for summarization.
    Determines the source (search results or history) and formats accordingly.

    Returns:
        A tuple: (formatted_content, source_type)
        source_type can be 'search_results', 'message_history', or None.
    """
    query = state.get("query", "").lower() # Get query for intent check
    search_results = state.get("search_results")
    messages = state.get("messages", []) # History + current query message
    content_to_summarize = ""
    source_type = None

    # Check if the user explicitly asked for a conversation summary
    is_history_summary_request = "conversation" in query or "history" in query or "previous" in query or "earlier" in query

    # Prioritize search results UNLESS it's explicitly a history summary request
    if search_results and isinstance(search_results, list) and not is_history_summary_request:
        valid_results = [res for res in search_results if isinstance(res, dict) and 'error' not in res and res.get('snippet')]
        if valid_results:
            logger.info(f"Summarizer: Found {len(valid_results)} valid search results to summarize.")
            source_type = "search_results"
            # Format results, clearly indicating multiple sources if present
            formatted_results = "---".join([f"Source {idx+1} (Title: {res.get('title', 'N/A')} | URL: {res.get('link', 'N/A')}):
{res.get('snippet', 'N/A')}" # Changed 'url' to 'link' based on previous context
                for idx, res in enumerate(valid_results)
            ])
            content_to_summarize = f"The following information was retrieved from web search results:
{formatted_results}"
        else:
             logger.warning("Summarizer: Search results found, but they seem empty, lack snippets, or contain errors. Checking history.")
             # Fall through

    # Use message history if no valid search results OR if history summary was requested
    # Exclude the *very last* message, which is the user's current request (e.g., "summarize this")
    relevant_history = messages[:-1]
    if not content_to_summarize and relevant_history:
        logger.info("Summarizer: Using message history for summarization.")
        source_type = "message_history"
        # Format the relevant history messages
        history_str = "
".join([f"{msg.type}: {msg.content}" for msg in relevant_history if isinstance(msg, BaseMessage)])
        content_to_summarize = f"The following is from the conversation history:
{history_str}"

    if not content_to_summarize:
        logger.warning("Summarizer: No suitable content found to summarize (neither search nor history).")
        return None, None

    return content_to_summarize, source_type


def summarize_node(state: GraphState) -> dict:
    """
    Condenses content (search results or message history) using an LLM.
    Aims to extract key information, potentially synthesizing from multiple sources if provided.
    Updates the 'summary' field in the state.
    """
    logger.info("--- Executing Summarizer Agent Node ---")
    summary_update = {"summary": None, "error": None}

    # 1. Extract and Format Content
    formatted_content, source_type = format_content_for_summary(state)

    if not formatted_content or not source_type:
        summary_update["summary"] = "No specific content was available to summarize at this step."
        logger.warning("Summarizer Node: Aborting summary due to lack of content or source type.")
        return summary_update

    # 2. Define Prompt based on Source Type and Content
    # This prompt is enhanced to guide the LLM better based on requirements
    prompt_template_str = """You are an expert Content Distiller agent. Your task is to process the provided text and generate a concise, informative summary tailored to the user's likely needs.

**Original User Query Context (The reason this summary was likely requested):**
{query}

**Content Source Type:** {source_type}

**Content to Process:**
{context}

**Instructions:**
1.  **Identify the Core Topic/Question:** What is the main subject or implicit question being addressed in the 'Content to Process'?
2.  **Extract Key Information:** Identify the most critical facts, findings, arguments, dates, names, or metrics relevant to the core topic and the original user query context.
3.  **Synthesize (if multiple sources):** If the content comes from multiple search results, synthesize the information. Note similarities, differences, or complementary points if applicable.
4.  **Format the Output:**
    *   Start with a brief introductory sentence stating the main topic.
    *   Use **bullet points** for the key extracted information for clarity and readability.
    *   If the original request seemed simple (like asking for a quick update), you *could* optionally add a **TL;DR** (Too Long; Didn't Read) sentence at the very end summarizing the absolute main point.
    *   Focus on neutrality and accuracy.
5.  **Keep it Concise:** The summary should be significantly shorter than the original content while retaining the essential information.

**Generate the summary below:**
Summary:
"""

    provider = state.get("llm_provider", "openai") # Use a capable model
    model = state.get("llm_model")

    try:
        llm = get_llm(provider=provider, model=model)
    except Exception as e:
        logger.error(f"Summarizer Node: Failed to get LLM instance - {e}", exc_info=True)
        return {"summary": f"Error: Could not initialize LLM.", "error": str(e)}

    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        logger.info(f"Summarizer Node: Invoking LLM ({provider}/{model or 'default'}) for summarization.")
        original_query = state.get('query', 'User requested a summary') # The query that *led* to summarization
        
        summary_result = chain.invoke({
            "query": original_query,
            "source_type": source_type,
            "context": formatted_content
        })

        if isinstance(summary_result, str) and summary_result:
            # Remove the leading "Summary:" if the LLM includes it
            final_summary = summary_result.strip()
            if final_summary.lower().startswith("summary:"):
                final_summary = final_summary[len("summary:"):].strip()
                
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
    return summary_update