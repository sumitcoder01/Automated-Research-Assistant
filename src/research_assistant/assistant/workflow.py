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
    workflow.add_node("assistant_analyze_route", assistant_agent.analyze_and_route_node)
    workflow.add_node("search", search_agent.search_node)
    workflow.add_node("summarize", summarizer_agent.summarize_node) # Single summarizer node
    workflow.add_node("generate_direct_response", assistant_agent.generate_direct_response_node)
    workflow.add_node("synthesize_response", assistant_agent.synthesize_response_node)

    # --- Define Control Flow ---
    workflow.set_entry_point("assistant_analyze_route")

    # --- Conditional Edges from Analysis/Routing ---
    workflow.add_conditional_edges(
        "assistant_analyze_route",
        lambda state: state.get("next_step"),
        {
            "needs_search": "search",
            "generate_direct_response": "generate_direct_response",
            "summarize_history": "summarize", # NEW: Route history summary requests to summarize
            "needs_summary": "summarize", # Route content summary requests to summarize
            "needs_complex_processing": "search",
            # Fallback could go to search or direct response? Let's keep search.
        },
        default="search" # Default path if next_step is unexpected
    )

    # --- Standard Workflow Edges ---
    # After search, always go to summarize (content summary)
    workflow.add_edge("search", "summarize")

    # --- Conditional Edges AFTER Summarization ---
    # Decide whether to synthesize further or end (if it was just a history summary)
    def after_summary_router(state: GraphState):
        if state.get("summary_request_type") == "history":
            # History summary is the final response
            logger.debug("Routing from Summarize to END (History Summary)")
            return END
        else:
            # Content summary needs synthesis into a final response
            logger.debug("Routing from Summarize to Synthesize (Content Summary)")
            return "synthesize_response"

    workflow.add_conditional_edges(
        "summarize",
        after_summary_router # Use the new routing function
        # The mapping dict is implicitly handled by the function returning node names or END
    )

    # --- Edges Leading to END ---
    workflow.add_edge("generate_direct_response", END)
    workflow.add_edge("synthesize_response", END) # Synthesis is always final

    # Compile the graph
    try:
        graph_app = workflow.compile()
        logger.info("LangGraph workflow compiled successfully with history summary routing.")
        return graph_app
    except Exception as e:
        logger.error(f"Error compiling LangGraph workflow: {e}", exc_info=True)
        raise

# Create the graph instance
graph_app = create_graph()
