from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool

from .base import BaseAgent
from ..prompts.summarization import SUMMARIZATION_PROMPT

class SummarizationAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        super().__init__(
            name="summarization_agent",
            description="Agent responsible for summarizing documents and extracting key insights",
            llm_provider=llm_provider,
            model=model,
            tools=tools
        )
        
    def get_prompt(self) -> ChatPromptTemplate:
        return SUMMARIZATION_PROMPT
        
    def should_use(self, query: str) -> bool:
        """Determine if this agent should be used for the given query."""
        # Check if the query is asking for summarization
        summarization_keywords = [
            "summarize", "summary", "summarization",
            "summarise", "summarisation", "brief",
            "overview", "key points", "main points"
        ]
        return any(keyword in query.lower() for keyword in summarization_keywords)
        
    def plan(self, intermediate_steps: List[tuple[AgentAction, str]]) -> Union[AgentAction, AgentFinish]:
        """Plan the next action based on intermediate steps."""
        # If we have the document content, we can proceed with summarization
        if intermediate_steps:
            return AgentFinish(
                return_values={"summary": intermediate_steps[-1][1]},
                log="Summarization completed successfully"
            )
        return AgentAction(
            tool="get_document_content",
            tool_input={"query": "Please provide the document content to summarize"},
            log="Requesting document content for summarization"
        )
        
    def act(self, intermediate_steps: List[tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Execute the next action."""
        return self.plan(intermediate_steps) 