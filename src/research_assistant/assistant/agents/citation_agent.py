from typing import Dict, List
from langchain.schema import BaseMessage

from ...llms.provider import get_llm
from ...prompts.citation import CITATION_PROMPT

class CitationAgent:
    def __init__(self):
        self.llm = get_llm()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the citation phase of research.
        """
        analysis = state["analysis"]
        search_results = state["search_results"]
        
        # Generate citations for each analysis
        citations = await self._generate_citations(analysis, search_results)
        
        # Update state
        state["citations"] = citations
        
        return state
    
    async def _generate_citations(
        self,
        analysis: List[Dict],
        search_results: List[Dict]
    ) -> List[Dict]:
        """
        Generate citations using LLM.
        """
        citations = []
        
        for item in analysis:
            messages = [
                {"role": "system", "content": CITATION_PROMPT},
                {
                    "role": "user",
                    "content": f"Content: {item['content']}\n\nSearch Results:\n{self._format_search_results(search_results)}"
                }
            ]
            
            response = await self.llm.agenerate([messages])
            citation_text = response.generations[0][0].text.strip()
            
            # Parse citations from response
            parsed_citations = self._parse_citations(citation_text, search_results)
            citations.extend(parsed_citations)
        
        return citations
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """
        Format search results for citation generation.
        """
        formatted = []
        for i, result in enumerate(search_results, 1):
            formatted.append(f"{i}. Title: {result['title']}")
            formatted.append(f"   URL: {result['url']}")
            formatted.append(f"   Content: {result['content']}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _parse_citations(self, citation_text: str, search_results: List[Dict]) -> List[Dict]:
        """
        Parse citations from LLM response.
        """
        citations = []
        lines = citation_text.split("\n")
        
        for line in lines:
            if not line.strip():
                continue
                
            # Extract source and snippet
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
                
            source = parts[0].strip()
            snippet = parts[1].strip()
            
            # Find matching search result
            for result in search_results:
                if source.lower() in result["title"].lower():
                    citations.append({
                        "source": result["title"],
                        "url": result["url"],
                        "snippet": snippet
                    })
                    break
        
        return citations 