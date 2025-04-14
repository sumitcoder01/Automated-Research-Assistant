import logging
from langgraph.graph import StateGraph, END
from research_assistant.assistant.graph.state import GraphState
from research_assistant.agents import assistant_agent, search_agent, summarizer_agent

logger = logging.getLogger(__name__)

# --- Build the Graph ---
def create_graph():
    """Creates the LangGraph workflow with updated routing logic."""
    workflow = StateGraph(GraphState)

    # --- Add Nodes ---
    # Use the new function names from the refactored agents
    workflow.add_node("assistant_analyze_route", assistant_agent.analyze_and_route_node) # Renamed node
    workflow.add_node("search", search_agent.search_node)
    workflow.add_node("summarize", summarizer_agent.summarize_node)
    workflow.add_node("generate_direct_response", assistant_agent.generate_direct_response_node)
    workflow.add_node("synthesize_response", assistant_agent.synthesize_response_node) # Final response synthesis

    # --- Define Control Flow ---

    # Start with the assistant analyzing and routing the query
    workflow.set_entry_point("assistant_analyze_route")

    # --- Conditional Edges from Analysis/Routing ---
    # This node decides the next step based on user intent and context
    workflow.add_conditional_edges(
        "assistant_analyze_route", # Source node
        lambda state: state.get("next_step"), # Decision function based on state output
        {
            # Map 'next_step' values to target nodes
            "needs_search": "search",
            "generate_direct_response": "generate_direct_response",
            "needs_summary": "summarize", # Route directly if analysis decided summary is next
            "needs_complex_processing": "search", # Start complex tasks with search for now
        }
    )
    workflow.add_edge("search", "summarize")

    # After summarizing (or if routed directly to summarize), synthesize the final response
    workflow.add_edge("summarize", "synthesize_response")

    # --- Edges Leading to END ---
    # Direct response generation ends the flow
    workflow.add_edge("generate_direct_response", END)
    # Final synthesis of search/summary results ends the flow
    workflow.add_edge("synthesize_response", END)

    # Compile the graph
    # Add error handling during compilation if needed
    try:
        graph_app = workflow.compile()
        logger.info("LangGraph workflow compiled successfully with updated routing.")
        return graph_app
    except Exception as e:
        logger.error(f"Error compiling LangGraph workflow: {e}", exc_info=True)
        raise # Re-raise the exception to prevent using a potentially broken graph

# Create the graph instance when the module is loaded
graph_app = create_graph()