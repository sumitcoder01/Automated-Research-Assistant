from typing import Any, Dict, List, Optional, Union
import aiohttp
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool

from .base import BaseAgent
from ..config import settings
from ..prompts.web_search import WEB_SEARCH_PROMPT

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using Searx for real-time information"
    
    async def _arun(self, query: str) -> str:
        """Run the web search asynchronously."""
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "time_range": "year"
            }
            async with session.get(settings.SEARX_URL, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    return self._format_results(results.get("results", []))
                return f"Error performing web search: {response.status}"
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string."""
        formatted_results = []
        for result in results[:5]:  # Limit to top 5 results
            formatted_results.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {result.get('content', 'N/A')}\n"
                "---"
            )
        return "\n".join(formatted_results)

class WebSearchAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        web_search_tool = WebSearchTool()
        super().__init__(
            name="web_search_agent",
            description="Agent responsible for performing web searches and analyzing results",
            llm_provider=llm_provider,
            model=model,
            tools=[web_search_tool]
        )
        
    def get_prompt(self) -> ChatPromptTemplate:
        return WEB_SEARCH_PROMPT
        
    def should_use(self, query: str) -> bool:
        """Determine if this agent should be used for the given query."""
        # Check if the query requires real-time information
        search_keywords = [
            "latest", "current", "recent", "now",
            "today", "this year", "this month",
            "find", "search", "look up"
        ]
        return any(keyword in query.lower() for keyword in search_keywords)
        
    def plan(self, intermediate_steps: List[tuple[AgentAction, str]]) -> Union[AgentAction, AgentFinish]:
        """Plan the next action based on intermediate steps."""
        if not intermediate_steps:
            return AgentAction(
                tool="web_search",
                tool_input={"query": "Please provide a search query"},
                log="Initiating web search"
            )
            
        # Analyze search results and determine if more searches are needed
        search_results = intermediate_steps[-1][1]
        if "Error" in search_results:
            return AgentFinish(
                return_values={"error": search_results},
                log="Web search failed"
            )
            
        return AgentFinish(
            return_values={"search_results": search_results},
            log="Web search completed successfully"
        )
        
    def act(self, intermediate_steps: List[tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Execute the next action."""
        return self.plan(intermediate_steps) 