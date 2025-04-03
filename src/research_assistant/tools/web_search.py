import httpx
from research_assistant.config import settings
from langchain.tools import tool
import logging
from research_assistant.schemas.agent_io import SearchResult
from typing import List, Dict

logger = logging.getLogger(__name__)

@tool # LangChain tool decorator
def perform_web_search(query: str) -> List[Dict[str, str]]:
    """
    Performs a web search using a SearxNG instance and returns formatted results.

    Args:
        query: The search query string.

    Returns:
        A list of dictionaries, each containing 'title', 'url', and 'snippet' of a search result.
        Returns an empty list if the search fails or no results are found.
    """
    if not settings.searx_instance_url:
        logger.error("Searx instance URL is not configured.")
        return [{"error": "Search tool not configured."}]

    search_url = f"{settings.searx_instance_url}/search"
    params = {
        "q": query,
        "format": "json",  # Request JSON format
        "engines": "google,bing", # Example: specify engines (check your Searx config)
        # Add other params like language if needed: 'language': 'en'
    }
    headers = {"Accept": "application/json"}

    try:
        with httpx.Client(timeout=10.0) as client: # Sync client
            response = client.get(search_url, params=params, headers=headers)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

        data = response.json()
        results = data.get("results", [])

        formatted_results = []
        for result in results[:5]: # Limit to top 5 results
            formatted_results.append({
                "title": result.get("title", "No Title"),
                "url": result.get("url", "#"),
                "snippet": result.get("content", "No snippet available.") # Searx often uses 'content' for snippet
            })

        if not formatted_results:
            logger.warning(f"No search results found for query: {query}")
            return [{"message": "No results found."}]

        logger.info(f"Web search successful for query: '{query}', returned {len(formatted_results)} results.")
        # Return list of dicts, can be easily serialized/used
        return formatted_results

    except httpx.RequestError as e:
        logger.error(f"Error during web search request to {search_url}: {e}")
        return [{"error": f"Search request failed: {e}"}]
    except httpx.HTTPStatusError as e:
        logger.error(f"Searx instance returned error status {e.response.status_code}: {e.response.text}")
        return [{"error": f"Search service error: Status {e.response.status_code}"}]
    except Exception as e:
        logger.error(f"An unexpected error occurred during web search: {e}")
        return [{"error": f"Unexpected search error: {e}"}]


if __name__ == "__main__":
    test_query = "LangGraph applications"
    search_results = perform_web_search.invoke({"query": test_query})
    print(search_results)