from typing import Dict, List
from langchain.schema import BaseMessage
import uuid

from ...llms.provider import get_llm
from ...prompts.report import REPORT_PROMPT

class ReportAgent:
    def __init__(self):
        self.llm = get_llm()
        
    async def process(self, state: Dict) -> Dict:
        """
        Process the report generation phase of research.
        """
        query = state["query"]
        summary = state["summary"]
        analysis = state["analysis"]
        citations = state["citations"]
        
        # Generate final report
        report = await self._generate_report(query, summary, analysis, citations)
        
        # Update state
        state["report"] = report
        state["query_id"] = str(uuid.uuid4())
        
        return state
    
    async def _generate_report(
        self,
        query: str,
        summary: str,
        analysis: List[Dict],
        citations: List[Dict]
    ) -> Dict:
        """
        Generate the final research report using LLM.
        """
        # Format analysis for report
        formatted_analysis = self._format_analysis(analysis)
        
        # Format citations for report
        formatted_citations = self._format_citations(citations)
        
        messages = [
            {"role": "system", "content": REPORT_PROMPT},
            {
                "role": "user",
                "content": f"""Query: {query}

Summary:
{summary}

Analysis:
{formatted_analysis}

Citations:
{formatted_citations}"""
            }
        ]
        
        response = await self.llm.agenerate([messages])
        report_text = response.generations[0][0].text.strip()
        
        return {
            "content": report_text,
            "query": query,
            "summary": summary,
            "analysis": analysis,
            "citations": citations
        }
    
    def _format_analysis(self, analysis: List[Dict]) -> str:
        """
        Format analysis for report generation.
        """
        formatted = []
        for item in analysis:
            formatted.append(f"Category: {item['category']}")
            formatted.append(f"Content: {item['content']}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _format_citations(self, citations: List[Dict]) -> str:
        """
        Format citations for report generation.
        """
        formatted = []
        for citation in citations:
            formatted.append(f"Source: {citation['source']}")
            formatted.append(f"URL: {citation['url']}")
            formatted.append(f"Snippet: {citation['snippet']}")
            formatted.append("")
        return "\n".join(formatted) 