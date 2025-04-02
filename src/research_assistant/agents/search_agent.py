import logging
from research_assistant.assistant.tools.web_search import perform_web_search
from research_assistant.assistant.graph.state import GraphState

logger = logging.getLogger(__name__)

def search_node(state: GraphState) -> dict:
    """Node to perform web search based on the query."""
    logger.info("Executing Search Node")
    query = state.get("query")
    if not query:
        logger.warning("Search Node: No query found in state.")
        return {"search_results": [{"error": "No query provided for search."}]}

    print(f"Search Node: Performing web search for: {query}")
    # Invoke the web search tool (already decorated with @tool)
    results = perform_web_search.invoke({"query": query}) # Tools expect dict input

    # TODO: Implement feedback loop/refinement if results are poor
    # This might involve another LLM call to generate a better query based on initial results

    print(f"Search Node: Received {len(results)} results.")
    return {"search_results": results}