# src/research_assistant/assistant/agents/search_agent.py
import logging
from research_assistant.tools.web_search import perform_web_search
from research_assistant.assistant.graph.state import GraphState

logger = logging.getLogger(__name__)

def search_node(state: GraphState) -> dict:
    """Node to perform web search based on the query."""
    logger.info("--- Executing Search Node ---")
    # Use the refined search query if the assistant provided one, else use original
    query_to_search = state.get("search_query") or state.get("query")

    if not query_to_search:
        logger.warning("Search Node: No query found in state.")
        return {"search_results": [{"error": "No query provided for search."}]}

    logger.info(f"Search Node: Performing web search for: {query_to_search}")
    results = perform_web_search.invoke({"query": query_to_search})

    logger.info(f"Search Node: Received {len(results)} results.")
    # Ensure results are added to the state
    return {"search_results": results}