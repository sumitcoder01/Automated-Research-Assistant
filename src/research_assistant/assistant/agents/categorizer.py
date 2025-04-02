from typing import Dict, List
from langchain.schema import BaseMessage

from ...llms.provider import get_llm
from ...prompts.categorization import CATEGORIZATION_PROMPT

class CategorizerAgent:
    def __init__(self):
        self.llm = get_llm()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the categorization phase of research.
        """
        summary = state["summary"]
        query = state["query"]
        
        # Generate categories
        categories = await self._generate_categories(summary, query)
        
        # Update state
        state["categories"] = categories
        
        return state
    
    async def _generate_categories(self, summary: str, query: str) -> List[str]:
        """
        Generate categories using LLM.
        """
        messages = [
            {"role": "system", "content": CATEGORIZATION_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nSummary:\n{summary}"}
        ]
        
        response = await self.llm.agenerate([messages])
        categories_text = response.generations[0][0].text.strip()
        
        # Parse categories from response
        categories = [cat.strip() for cat in categories_text.split("\n") if cat.strip()]
        return categories 