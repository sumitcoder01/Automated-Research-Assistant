from fastapi import APIRouter, Depends, HTTPException, status
import logging
import json
from typing import List
from research_assistant.schemas.query import QueryRequest, QueryResponse
from research_assistant.api.deps import get_session_store, BaseSessionStore
from research_assistant.assistant.workflow import graph_app
from research_assistant.assistant.graph.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage 

router = APIRouter(tags=["Research Query"]) 
logger = logging.getLogger(__name__)

@router.post("", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    store: BaseSessionStore = Depends(get_session_store)
):
    """
    Handles a user query, orchestrates agents via LangGraph, and returns the response.
    Uses the specified provider for embeddings when adding messages via the injected store.
    Loads previous conversation history.
    """
    session_id = request.session_id
    user_query = request.query
    embedding_provider_for_session = request.embedding_provider or "google" # Default

    logger.info(f"Received query for session {session_id} (LLM: {request.llm_provider}, Embeddings: {embedding_provider_for_session}): '{user_query[:50]}...'")

    try:
        # --- STEP 1: RETRIEVE HISTORY ---

        history_limit = 10 # Adjust as needed based on token limits and context needs
        logger.debug(f"Retrieving history for session {session_id} with limit {history_limit}")
        history: List[BaseMessage] = store.get_history(session_id, limit=history_limit)
        logger.debug(f"Retrieved {len(history)} messages from history.")

        # --- STEP 2: Prepare current user message ---
        user_message = HumanMessage(content=user_query)

        # --- STEP 3: Add user message to store (Do this AFTER retrieving history for the current turn) ---
        store.add_message(
            session_id,
            user_message,
            embedding_provider=embedding_provider_for_session
        )

        # --- STEP 4: Prepare initial state for the graph INCLUDING HISTORY ---
        # Combine history with the current message
        current_conversation: List[BaseMessage] = history + [user_message]

        initial_state: GraphState = {
            "query": user_query, # Keep the original query separate for clarity if needed
            "session_id": session_id,
            "llm_provider": request.llm_provider or "deepseek", # Default LLM provider
            "llm_model": request.llm_model,
            "embedding_provider": embedding_provider_for_session,
            "messages": current_conversation, # Pass the combined history + current message
            # Initialize other state fields
            "search_query": None,
            "search_results": None,
            "summary": None,
            "final_response": None,
            "next_step": None,
            "error": None
        }
        logger.debug(f"Initial graph state includes {len(current_conversation)} messages.")


        # --- STEP 5: Invoke the LangGraph workflow ---
        graph_config = {"configurable": {"session_id": session_id}}
        final_state = await graph_app.ainvoke(initial_state, config=graph_config)

        if not isinstance(final_state, dict):
             logger.error(f"Graph execution returned unexpected type: {type(final_state)}")
             try: # Best effort error logging to store
                error_content = f"Sorry, an internal error occurred: Graph returned invalid type {type(final_state).__name__}."
                store.add_message(session_id, AIMessage(content=error_content), embedding_provider=embedding_provider_for_session)
             except Exception as store_e:
                 logger.error(f"Failed to store graph type error message for session {session_id}: {store_e}")
             raise HTTPException(status_code=500, detail="Internal error during graph execution")

        logger.info(f"Graph execution completed for session {session_id}.")

        # --- STEP 6: Determine the final response from the state ---

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
        elif not final_state.get("error"):
             response_content = "Okay."


        # --- STEP 7: Add assistant response to store ---

        assistant_message = AIMessage(content=response_content)
        store.add_message(
            session_id,
            assistant_message,
            embedding_provider=embedding_provider_for_session
        )

        # --- STEP 8: Return response ---

        return QueryResponse(
            session_id=session_id,
            query=user_query,
            response=response_content,
        )

    except ValueError as ve:
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