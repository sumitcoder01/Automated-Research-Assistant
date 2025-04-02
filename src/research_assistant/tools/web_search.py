from typing import List, Dict
import httpx
from bs4 import BeautifulSoup
import json

from ...config import settings

class WebSearchTool:
    def __init__(self):
        self.searx_url = settings.SEARX_URL
        self.client = httpx.AsyncClient()
        
    async def search(self, query: str) -> List[Dict]:
        """
        Perform a web search using Searx.
        """
        try:
            # Construct search URL
            search_url = f"{self.searx_url}/search"
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "max_results": 10
            }
            
            # Make request
            response = await self.client.get(search_url, params=params)
            response.raise_for_status()
            
            # Parse results
            results = response.json()
            return self._process_results(results)
            
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return []
            
    def _process_results(self, results: Dict) -> List[Dict]:
        """
        Process and clean search results.
        """
        processed_results = []
        
        for result in results.get("results", []):
            # Extract content
            content = self._extract_content(result.get("content", ""))
            
            processed_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content
            })
            
        return processed_results
    
    def _extract_content(self, html_content: str) -> str:
        """
        Extract clean text content from HTML.
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return html_content
    
    async def close(self):
        """
        Close the HTTP client.
        """
        await self.client.aclose() 