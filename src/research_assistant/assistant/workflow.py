import logging
from langgraph.graph import StateGraph, END
from research_assistant.assistant.graph.state import GraphState
from research_assistant.assistant.agents import summarizer, search_agent
from research_assistant.llms.provider import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# --- Agent Nodes ---
# (Imported from agents directory)
# e.g., from .agents.summarizer import summarize_node

# --- Tool Nodes ---
# Tools are usually called *within* agent nodes or via specific tool nodes if needed.
# Here, search_agent.search_node calls the tool directly.

# --- Router Logic ---
def route_query(state: GraphState) -> str:
    """
    Determines the next step based on the user query or current state.
    This is a crucial LLM-driven decision point.
    """
    logger.info("Executing Router Node")
    query = state['query'].lower()
    messages = state.get('messages', [])
    last_message = messages[-1].content.lower() if messages else query

    # Simple keyword routing (Placeholder - Replace with LLM call)
    if "summarize" in last_message:
         print("Routing decided: Summarize")
         return "summarize"
    elif any(kw in last_message for kw in ["search", "find", "look up", "what is", "who is"]):
        print("Routing decided: Search")
        return "search"
    elif any(kw in last_message for kw in ["hi", "hello", "hey"]):
         print("Routing decided: Greet/End")
         # For simple greetings, we might just end or have a dedicated greet node
         return END # Or "greet_node" if you add one
    else:
         # Default action if unsure - perhaps search or ask for clarification
         print("Routing decided: Defaulting to Search (or could be END/Clarify)")
         # return "search" # Decided against default search for now
         return END # Let's end if no clear action

    # --- LLM-based routing (Conceptual) ---
    # llm = get_llm(state.get('llm_provider', 'openai'))
    # routing_prompt = f"""Given the user query: '{query}', decide the primary next action.
    # Options are: 'search', 'summarize', 'categorize', 'analyze', 'report', 'greet', 'end'.
    # Query: {last_message}
    # Decision:"""
    # response = llm.invoke(routing_prompt)
    # next_node = response.content.strip().lower()
    # logger.info(f"LLM Router decided: {next_node}")
    # if next_node in ["search", "summarize", ...]: # Validate response
    #     return next_node
    # return END # Default to end if LLM fails or gives invalid response

# --- Build the Graph ---
def create_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("search", search_agent.search_node)
    workflow.add_node("summarize", summarizer.summarize_node)
    # workflow.add_node("categorize", categorize_node) # Add other nodes
    # workflow.add_node("analyze", analyze_node)
    # workflow.add_node("report", report_node)

    # Entry Point
    workflow.set_entry_point("router") # Start with the router deciding the first step

    # Conditional Entry Router
    # The router node itself needs to be added. It calls the route_query function.
    # We can make the router function the node directly if its signature matches.
    workflow.add_node("router", route_query)


    # Define Edges based on Router Logic
    # The router decides where to go *first*. Subsequent steps might be linear or also conditional.
    workflow.add_conditional_edges(
        "router", # Source node
        lambda state: state.get("next_node") or route_query(state), # Function to determine the next node dynamically
         {
             "search": "search",
             "summarize": "summarize",
             # Add mappings for other agent nodes
             # "categorize": "categorize",
             END: END # Map the decision 'END' to the actual END state
         }
    )

    # Define transitions *after* specific nodes (Example: after search, maybe summarize)
    # This part defines the more complex workflow beyond the initial routing.
    # Option 1: Linear flow after search
    # workflow.add_edge("search", "summarize")
    # workflow.add_edge("summarize", END) # Example end

    # Option 2: Conditional flow after search (e.g., summarize only if results exist)
    def should_summarize(state: GraphState) -> str:
        if state.get("search_results") and not state.get("summary"): # Check if search ran and no summary yet
             print("Decision: Summarize after search")
             return "summarize"
        else:
             print("Decision: End after search (no results or already summarized)")
             return END

    workflow.add_conditional_edges("search", should_summarize, {
         "summarize": "summarize",
         END: END
    })

    # Define what happens after summarize (usually END in this simple flow)
    workflow.add_edge("summarize", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return app