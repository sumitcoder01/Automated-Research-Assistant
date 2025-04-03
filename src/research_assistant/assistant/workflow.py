# src/research_assistant/assistant/graph/workflow.py
import logging
from langgraph.graph import StateGraph, END
from research_assistant.assistant.graph.state import GraphState
from research_assistant.agents import assistant_agent, search_agent, summarizer_agent

logger = logging.getLogger(__name__)

# --- Build the Graph ---
def create_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes from agent files
    workflow.add_node("assistant_analyze", assistant_agent.analyze_query_node)
    workflow.add_node("search", search_agent.search_node)
    workflow.add_node("summarize", summarizer_agent.summarize_node)
    workflow.add_node("generate_direct_response", assistant_agent.generate_direct_response_node)
    workflow.add_node("synthesize_response", assistant_agent.synthesize_response_node)

    # --- Define Control Flow ---

    # Start with the assistant analyzing the query
    workflow.set_entry_point("assistant_analyze")

    # Conditional edge after analysis
    workflow.add_conditional_edges(
        "assistant_analyze", # Source node
        lambda state: state.get("next_step"), # Decision made by the node
        {
            "needs_search": "search",
            "generate_direct_response": "generate_direct_response",
            # Add mapping for potential error or fallback if needed
        }
    )

    # Path after search -> summarize
    workflow.add_edge("search", "summarize")

    # Path after summarize -> synthesize response
    workflow.add_edge("summarize", "synthesize_response")

    # Edges leading to END
    workflow.add_edge("generate_direct_response", END)
    workflow.add_edge("synthesize_response", END)

    # Compile the graph
    graph_app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return graph_app

# Create the instance at module level
graph_app = create_graph()
