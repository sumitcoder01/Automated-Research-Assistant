from typing import Dict, List, Optional, Tuple, Any
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain.schema import AgentAction, AgentFinish

from .agents.base import BaseAgent
from .agents.summarization import SummarizationAgent
from .agents.web_search import WebSearchAgent
from .config import settings
from .memory import MemoryStore

class ResearchOrchestrator:
    def __init__(self):
        self.agents: List[BaseAgent] = [
            SummarizationAgent(),
            WebSearchAgent()
        ]
        self.memory = MemoryStore()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(StateType=Dict)
        
        # Add nodes for each agent
        for agent in self.agents:
            workflow.add_node(agent.name, self._create_agent_node(agent))
            
        # Add edges between agents
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                workflow.add_edge(agent1.name, agent2.name)
                
        # Set the entry point
        workflow.set_entry_point(self.agents[0].name)
        
        # Compile the graph
        return workflow.compile()
        
    def _create_agent_node(self, agent: BaseAgent):
        """Create a node function for an agent."""
        async def agent_node(state: Dict) -> Dict:
            # Check if this agent should handle the query
            if not agent.should_use(state["query"]):
                return state
                
            # Get the agent's executor
            executor = agent.get_executor()
            
            # Run the agent
            result = await executor.arun(
                input=state["query"],
                **state.get("agent_kwargs", {})
            )
            
            # Update state with results
            state["results"].append({
                "agent": agent.name,
                "result": result
            })
            
            return state
            
        return agent_node
        
    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Process a research query using the multi-agent system."""
        # Initialize or retrieve session
        if session_id:
            session = await self.memory.get_session(session_id)
        else:
            session = await self.memory.create_session()
            
        # Prepare initial state
        initial_state = {
            "query": query,
            "session_id": session.id,
            "results": [],
            "agent_kwargs": kwargs
        }
        
        # Run the workflow
        final_state = await self.graph.arun(initial_state)
        
        # Store results in memory
        await self.memory.add_interaction(
            session.id,
            query,
            final_state["results"]
        )
        
        return {
            "session_id": session.id,
            "results": final_state["results"]
        } 