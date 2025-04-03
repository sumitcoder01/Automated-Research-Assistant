from research_assistant.config import settings
from langchain.tools import tool
import logging
from typing import List, Dict, Union

# Import the Tavily client
from tavily import TavilyClient

logger = logging.getLogger(__name__)

@tool # LangChain tool decorator still works
def perform_web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using the Tavily Search API and returns formatted results.

    Args:
        query: The search query string.
        max_results: The maximum number of results to return.

    Returns:
        A list of dictionaries, each containing 'title', 'url', and 'snippet' of a search result.
        Returns a list containing an error dictionary if the search fails.
    """
    if not settings.tavily_api_key:
        logger.error("Tavily API key is not configured in settings.")
        # Return structure indicates error, consistent with potential API errors
        return [{"error": "Search tool not configured (Missing API Key)."}]

    try:
        # Initialize the Tavily client
        client = TavilyClient(api_key=settings.tavily_api_key)

    except Exception as e:
         logger.error(f"Failed to initialize Tavily client: {e}", exc_info=True)
         return [{"error": f"Search tool initialization failed: {e}"}]

    try:
        logger.info(f"Performing Tavily search for query: '{query}' (max_results={max_results})")
        
        # Execute the search query
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results
        )
        # Response structure is typically {'query': '...', 'results': [{'title': ..., 'url': ..., 'content': ...}, ...]}

    except Exception as e:
        logger.error(f"An error occurred during Tavily search for query '{query}': {e}", exc_info=True)
        # Try to provide a slightly more informative error message
        error_message = f"Tavily search failed: {type(e).__name__}"
        # Check common attributes for details, or just use the exception string
        details = getattr(e, 'message', str(e))
        if details:
             error_message += f" - {details}"
        return [{"error": error_message}]

    # Process the results
    results = response.get("results", [])

    if not results:
        logger.warning(f"No search results found by Tavily for query: {query}")
        # Return a message indicating no results, not an error
        return [{"message": "No results found."}]

    # Format the results into the desired structure
    formatted_results = []
    for result in results:
        formatted_results.append({
            "title": result.get("title", "No Title"),
            "url": result.get("url", "#"),
            # Map Tavily's 'content' field to 'snippet' for consistency if needed downstream
            "snippet": result.get("content", "No snippet available.")
        })

    logger.info(f"Tavily search successful for query: '{query}', returned {len(formatted_results)} results.")
    return formatted_results


# --- Optional: Update the test block for direct execution ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import json

    # Load .env file from the project root for testing
    # Adjust path if your execution context is different
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Create a minimal settings object for testing
    class MockSettings:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
    settings = MockSettings()

    if settings.tavily_api_key:
        test_query = "What are the latest developments in quantum computing?"
        print(f"Testing Tavily search with query: '{test_query}'")
        search_results = perform_web_search.invoke({"query": test_query})
        print("\nSearch Results:")
        print(json.dumps(search_results, indent=2))

        print("-" * 20)
        test_query_no_results = "asdlkfjaslkdfjhasdflkjh" # Unlikely to have results
        print(f"Testing Tavily search with query likely yielding no results: '{test_query_no_results}'")
        search_results_none = perform_web_search.invoke({"query": test_query_no_results})
        print("\nSearch Results (None Expected):")
        print(json.dumps(search_results_none, indent=2))
    else:
        print("TAVILY_API_KEY not found in environment variables or .env file. Skipping test.")
        print(f"Looked for .env at: {dotenv_path}")