from typing import List, Optional
from langgraph.graph import Graph
from langchain.schema import BaseMessage

from ..memory.chroma_store import ChromaStore
from .agents import (
    SearchAgent,
    SummarizerAgent,
    CategorizerAgent,
    AnalysisAgent,
    CitationAgent,
    ReportAgent
)
from ..schemas.query import QueryResponse, Finding, Citation

class ResearchWorkflow:
    def __init__(self, memory_store: ChromaStore):
        self.memory_store = memory_store
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Graph:
        """
        Build the LangGraph workflow for research processing.
        """
        # Initialize agents
        search_agent = SearchAgent()
        summarizer = SummarizerAgent()
        categorizer = CategorizerAgent()
        analyzer = AnalysisAgent()
        citation_agent = CitationAgent()
        report_agent = ReportAgent()
        
        # Define the graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("search", search_agent.process)
        workflow.add_node("summarize", summarizer.process)
        workflow.add_node("categorize", categorizer.process)
        workflow.add_node("analyze", analyzer.process)
        workflow.add_node("cite", citation_agent.process)
        workflow.add_node("report", report_agent.process)
        
        # Define edges and conditions
        workflow.add_edge("search", "summarize")
        workflow.add_edge("summarize", "categorize")
        workflow.add_edge("categorize", "analyze")
        workflow.add_edge("analyze", "cite")
        workflow.add_edge("cite", "report")
        
        # Set entry and exit points
        workflow.set_entry_point("search")
        workflow.set_exit_point("report")
        
        return workflow.compile()
    
    async def process_query(self, query: str) -> QueryResponse:
        """
        Process a research query through the workflow.
        """
        # Initialize state
        state = {
            "query": query,
            "search_results": [],
            "summary": "",
            "categories": [],
            "analysis": [],
            "citations": [],
            "report": None
        }
        
        # Run the workflow
        result = await self.graph.arun(state)
        
        # Store in memory
        await self.memory_store.store_query(query, result)
        
        return QueryResponse(
            query_id=result["query_id"],
            summary=result["summary"],
            findings=result["findings"],
            citations=result["citations"]
        )
    
    def cleanup(self):
        """
        Clean up resources.
        """
        self.graph = None 