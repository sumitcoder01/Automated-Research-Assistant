from typing import Any, Dict, List, Optional, Union
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool

from ..llm.providers import LLMProviderManager
from ..config import settings

class BaseAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        self.name = name
        self.description = description
        self.llm = LLMProviderManager.get_llm(llm_provider, model)
        self.tools = tools or []
        
    def get_prompt(self) -> ChatPromptTemplate:
        """Get the agent's prompt template. Should be implemented by subclasses."""
        raise NotImplementedError
        
    def get_executor(self) -> AgentExecutor:
        """Create and return an AgentExecutor instance."""
        return AgentExecutor.from_agent_and_tools(
            agent=self,
            tools=self.tools,
            llm=self.llm,
            verbose=settings.DEBUG,
            handle_parsing_errors=True
        )
        
    def plan(self, intermediate_steps: List[tuple[AgentAction, str]]) -> Union[AgentAction, AgentFinish]:
        """Plan the next action based on intermediate steps."""
        raise NotImplementedError
        
    def act(self, intermediate_steps: List[tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Execute the next action."""
        raise NotImplementedError
        
    def should_use(self, query: str) -> bool:
        """Determine if this agent should be used for the given query."""
        raise NotImplementedError 