from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    # Input/Config
    query: str
    session_id: str
    llm_provider: str
    llm_model: Optional[str]
    embedding_provider: str # Track for memory consistency

    # Conversation flow
    messages: List[BaseMessage] # Appended during the process

    # Agent Outputs / Intermediate results
    search_query: Optional[str] # Potentially refined query for search
    search_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    final_response: Optional[str] # The final generated answer for the user

    # Control Flow
    # Decide next step: 'generate_direct_response', 'needs_search', 'synthesize_response'
    next_step: Optional[str]
    error: Optional[str]