import logging
from research_assistant.tools.web_search import perform_web_search
from research_assistant.assistant.graph.state import GraphState

logger = logging.getLogger(__name__)

def search_node(state: GraphState) -> dict:
    """
    Node to perform web search using external tools based on the search query.
    Retrieves structured snippets (URL, title, snippet) for further processing.
    """
    logger.info("--- Executing Search Agent Node ---")
    # Use the refined search query if the assistant provided one, else use original query
    query_to_search = state.get("search_query", state.get("query")) # Get search_query, fallback to query

    if not query_to_search:
        logger.warning("Search Node: No query found in state.")
        # Return an empty list or an error structure consistent with successful results
        return {"search_results": []} # Return empty list for consistency

    logger.info(f"Search Node: Performing web search for: '{query_to_search}'")

    try:
        results = perform_web_search.invoke({"query": query_to_search})

        # Log the number of results received
        if isinstance(results, list):
             logger.info(f"Search Node: Received {len(results)} results.")
        else:
             logger.warning(f"Search Node: Received unexpected result format: {type(results)}. Expected list.")
             results = [] 

        # Return the structured results
        return {"search_results": results}

    except Exception as e:
        logger.error(f"Error during web search for '{query_to_search}': {e}", exc_info=True)
        # Return an empty list or error indicator in the standard format
        return {"search_results": [{"error": f"Failed to perform search: {e}"}]}