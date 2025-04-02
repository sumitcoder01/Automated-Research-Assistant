from typing import Dict, List
from langchain.schema import BaseMessage

from ...llms.provider import get_llm
from ...prompts.analysis import ANALYSIS_PROMPT

class AnalysisAgent:
    def __init__(self):
        self.llm = get_llm()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the analysis phase of research.
        """
        summary = state["summary"]
        categories = state["categories"]
        query = state["query"]
        
        # Generate analysis for each category
        analysis = await self._generate_analysis(summary, categories, query)
        
        # Update state
        state["analysis"] = analysis
        
        return state
    
    async def _generate_analysis(
        self,
        summary: str,
        categories: List[str],
        query: str
    ) -> List[Dict]:
        """
        Generate analysis for each category using LLM.
        """
        analysis = []
        
        for category in categories:
            messages = [
                {"role": "system", "content": ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": f"Query: {query}\nCategory: {category}\n\nSummary:\n{summary}"
                }
            ]
            
            response = await self.llm.agenerate([messages])
            content = response.generations[0][0].text.strip()
            
            analysis.append({
                "category": category,
                "content": content
            })
        
        return analysis 