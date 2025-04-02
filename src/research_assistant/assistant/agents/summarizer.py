from typing import Dict, List
from langchain.schema import BaseMessage

from ...llms.provider import get_llm
from ...prompts.summarization import SUMMARIZATION_PROMPT

class SummarizerAgent:
    def __init__(self):
        self.llm = get_llm()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the summarization phase of research.
        """
        search_results = state["search_results"]
        query = state["query"]
        
        # Prepare content for summarization
        content = self._prepare_content(search_results)
        
        # Generate summary
        summary = await self._generate_summary(content, query)
        
        # Update state
        state["summary"] = summary
        
        return state
    
    def _prepare_content(self, search_results: List[Dict]) -> str:
        """
        Prepare search results for summarization.
        """
        content = []
        for result in search_results:
            content.append(f"Title: {result['title']}")
            content.append(f"Content: {result['content']}")
            content.append("---")
        return "\n".join(content)
    
    async def _generate_summary(self, content: str, query: str) -> str:
        """
        Generate a summary using LLM.
        """
        messages = [
            {"role": "system", "content": SUMMARIZATION_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nContent:\n{content}"}
        ]
        
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text.strip() 