from typing import Dict, List
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage

from ...tools.web_search import WebSearchTool
from ...llms.provider import get_llm
from ...prompts.search_refinement import SEARCH_REFINEMENT_PROMPT

class SearchAgent:
    def __init__(self):
        self.llm = get_llm()
        self.search_tool = WebSearchTool()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the search phase of research.
        """
        query = state["query"]
        
        # Refine search query
        refined_query = await self._refine_query(query)
        
        # Perform web search
        search_results = await self.search_tool.search(refined_query)
        
        # Update state
        state["search_results"] = search_results
        state["refined_query"] = refined_query
        
        return state
    
    async def _refine_query(self, query: str) -> str:
        """
        Refine the search query using LLM.
        """
        messages = [
            {"role": "system", "content": SEARCH_REFINEMENT_PROMPT},
            {"role": "user", "content": query}
        ]
        
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text.strip() 