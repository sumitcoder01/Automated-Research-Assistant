from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The initial user query.
        session_id: The unique ID for the current session.
        llm_provider: The selected LLM provider.
        llm_model: The specific LLM model.
        messages: List of messages in the current conversation flow.
        search_results: Results from the web search tool.
        summary: Generated summary of content.
        category: Assigned category for the research topic.
        analysis: Results of analysis agent.
        report: The final generated report content.
        error: Optional error message if a node fails.
        next_node: Hint for the router (optional).
    """
    query: str
    session_id: str
    llm_provider: str
    llm_model: Optional[str]

    # Conversation history or intermediate messages for the current run
    messages: List[BaseMessage]

    # Agent outputs
    search_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    category: Optional[str]
    analysis: Optional[str]
    report: Optional[str]

    # Control flow and errors
    error: Optional[str]
    next_node: Optional[str] # Explicit hint for next step if needed