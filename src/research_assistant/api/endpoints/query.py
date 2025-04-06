from fastapi import APIRouter, Depends, HTTPException, status
import logging
import json
from research_assistant.schemas.query import QueryRequest, QueryResponse
from research_assistant.api.deps import get_session_store, BaseSessionStore
from research_assistant.assistant.workflow import graph_app
from research_assistant.assistant.graph.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter(tags=["Research Query"]) 
logger = logging.getLogger(__name__)

# Use "" for path relative to prefix in main.py
@router.post("", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    # Use the BaseSessionStore type hint for the injected dependency
    store: BaseSessionStore = Depends(get_session_store)
):
    """
    Handles a user query, orchestrates agents via LangGraph, and returns the response.
    Uses the specified provider for embeddings when adding messages via the injected store.
    """
    session_id = request.session_id
    user_query = request.query
    # Determine the embedding provider based on the llm_provider field (or a dedicated field)
    embedding_provider_for_session = request.embedding_provider or "google" # Default

    logger.info(f"Received query for session {session_id} (LLM: {request.llm_provider}, Embeddings: {embedding_provider_for_session}): '{user_query[:50]}...'")

    try:
        # 1. Retrieve history (optional - not currently used but could be added)
        # history = store.get_history(session_id, limit=10) # Example

        # 2. Add user message to store, PASSING THE PROVIDER HINT
        # This call works on any store implementing the BaseSessionStore interface
        user_message = HumanMessage(content=user_query)
        store.add_message(
            session_id,
            user_message,
            embedding_provider=embedding_provider_for_session
        )

        # 3. Prepare initial state for the graph
        initial_state: GraphState = {
            "query": user_query,
            "session_id": session_id,
            "llm_provider": request.llm_provider or "deepseek", # Default LLM provider for graph
            "llm_model": request.llm_model,
            "embedding_provider": embedding_provider_for_session, # Pass this along if needed in graph
            "messages": [user_message],
            # Initialize other state fields
            "search_query": None,
            "search_results": None,
            "summary": None,
            "final_response": None,
            "next_step": None,
            "error": None
        }

        # 4. Invoke the LangGraph workflow
        graph_config = {"configurable": {"session_id": session_id}}
        final_state = await graph_app.ainvoke(initial_state, config=graph_config)

        # Defensive check for graph output type
        if not isinstance(final_state, dict):
             logger.error(f"Graph execution returned unexpected type: {type(final_state)}")
             # Attempt to store error message before raising HTTP exception
             try:
                error_content = f"Sorry, an internal error occurred: Graph returned invalid type {type(final_state).__name__}."
                store.add_message(session_id, AIMessage(content=error_content), embedding_provider=embedding_provider_for_session)
             except Exception as store_e:
                 logger.error(f"Failed to store graph type error message for session {session_id}: {store_e}")
             raise HTTPException(status_code=500, detail="Internal error during graph execution")

        logger.info(f"Graph execution completed for session {session_id}.")
        # (Optional: Safe logging of final state using json.dumps with default handler)

        # 5. Determine the final response from the state
        # (Your existing logic for determining response_content is good)
        response_content = "Processing complete, but no final response was generated." # Default
        if final_state.get("error"):
            response_content = f"An error occurred: {final_state['error']}"
        elif final_state.get("final_response"):
            response_content = final_state["final_response"]
        elif final_state.get("summary"):
             response_content = "Summary generated:\n" + final_state["summary"]
        elif final_state.get("search_results"):
             valid_results = [r for r in final_state.get('search_results', []) if isinstance(r,dict) and 'error' not in r]
             if valid_results:
                response_content = f"Search completed with {len(valid_results)} results. Synthesis step may have failed or was not applicable."
             else:
                response_content = "Search was attempted but yielded no valid results or failed."
        # Add a fallback if no specific output found but graph didn't error
        elif not final_state.get("error"):
             response_content = "Okay." # Simple ack if graph finished without specific output


        # 6. Add assistant response to store
        # This call also works on any store implementing the BaseSessionStore interface
        assistant_message = AIMessage(content=response_content)
        store.add_message(
            session_id,
            assistant_message,
            embedding_provider=embedding_provider_for_session
        )

        # 7. Return response
        return QueryResponse(
            session_id=session_id,
            query=user_query,
            response=response_content,
        )

    except ValueError as ve: # Catch config errors like missing API keys from get_embedding_function
         logger.error(f"Configuration error handling query for session {session_id}: {ve}")
         # Add error message to store before raising
         try:
             error_content = f"Sorry, a configuration error occurred: {ve}"
             store.add_message(session_id, AIMessage(content=error_content), embedding_provider=embedding_provider_for_session)
         except Exception as store_e:
             logger.error(f"Failed to store config error message for session {session_id}: {store_e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Configuration error: {ve}")

    except Exception as e:
        logger.exception(f"Unexpected error handling query for session {session_id}: {e}")
        # Attempt to add error message to history (best effort)
        try:
             error_content = f"Sorry, an internal error occurred processing your request."
             store.add_message(session_id, AIMessage(content=error_content), embedding_provider=embedding_provider_for_session)
        except Exception as store_e:
             logger.error(f"Failed to store unexpected error message for session {session_id}: {store_e}")

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred.")