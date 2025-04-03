from fastapi import APIRouter, Depends, HTTPException, status
from research_assistant.schemas.query import QueryRequest, QueryResponse
from research_assistant.memory import ChromaSessionStore
from research_assistant.api.deps import get_session_store
from research_assistant.assistant.workflow import graph_app # Import compiled graph
from research_assistant.assistant.graph.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json # For safe printing of potentially complex state

router = APIRouter(tags=["Research Query"])
logger = logging.getLogger(__name__)

@router.post("/", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    store: ChromaSessionStore = Depends(get_session_store)
):
    """
    Handles a user query, orchestrates agents via LangGraph, and returns the response.
    """
    session_id = request.session_id
    user_query = request.query
    logger.info(f"Received query for session {session_id}: '{user_query[:50]}...'")

    try:
        # 1. Retrieve history (optional, but good for context)
        # history = store.get_history(session_id, limit=10) # Get recent messages
        # Format history for the graph if needed

        # 2. Add user message to store
        user_message = HumanMessage(content=user_query)
        store.add_message(session_id, user_message)

        # 3. Prepare initial state for the graph
        initial_state: GraphState = {
            "query": user_query,
            "session_id": session_id,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
            "messages": [user_message], # Start with the current query
            # Initialize other state fields as empty/None
            "search_results": None,
            "summary": None,
            "category": None,
            "analysis": None,
            "report": None,
            "error": None,
            "next_node": None,
        }

        # 4. Invoke the LangGraph workflow
        # The `configurable` dict is important for passing runtime info,
        # especially useful for Langsmith tracing correlation.
        graph_config = {"configurable": {"session_id": session_id}}
        final_state = await graph_app.ainvoke(initial_state, config=graph_config) # Use ainvoke for async FastAPI

        # Defensive check for final_state type
        if not isinstance(final_state, dict):
             logger.error(f"Graph execution returned unexpected type: {type(final_state)}")
             raise HTTPException(status_code=500, detail="Internal error during graph execution")


        logger.info(f"Graph execution completed for session {session_id}.")
        # Safely log final state (avoiding excessive length or sensitive data if necessary)
        try:
            final_state_str = json.dumps({k: (str(v)[:200] + '...' if isinstance(v, (str, list, dict)) and len(str(v)) > 200 else v) for k, v in final_state.items()}, indent=2)
            logger.debug(f"Final state for session {session_id}:\n{final_state_str}")
        except Exception as log_e:
            logger.warning(f"Could not serialize final state for logging: {log_e}")


        # 5. Determine the final response from the state
        response_content = "Could not determine a final response." # Default
        if final_state.get("error"):
            response_content = f"An error occurred: {final_state['error']}"
        elif final_state.get("report"):
            response_content = final_state["report"]
        elif final_state.get("summary"):
            response_content = final_state["summary"]
        elif final_state.get("search_results"):
             # Simple response if only search happened
             response_content = f"Found {len(final_state['search_results'])} results. You might want to ask for a summary."
        elif final_state.get("messages"):
             # Try to get the last AI message if available
             ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
             if ai_messages:
                response_content = ai_messages[-1].content
             else: # Fallback if no AI message (e.g., simple greet ended the graph)
                response_content = "Okay."


        # 6. Add assistant response to store
        assistant_message = AIMessage(content=response_content)
        store.add_message(session_id, assistant_message)

        # 7. Return response
        return QueryResponse(
            session_id=session_id,
            query=user_query,
            response=response_content,
            # Optionally include some debug info from final_state if needed
            # debug_info={"final_node": final_state.get("next_node", "END")}
        )

    except Exception as e:
        logger.exception(f"Error handling query for session {session_id}: {e}") # Log full traceback
        # Attempt to add error message to history
        try:
             error_message = AIMessage(content=f"Sorry, an internal error occurred: {e}")
             store.add_message(session_id, error_message)
        except Exception as store_e:
             logger.error(f"Failed to store error message for session {session_id}: {store_e}")

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {e}")