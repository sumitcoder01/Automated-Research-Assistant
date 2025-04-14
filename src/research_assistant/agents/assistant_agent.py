import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
# --- Hypothetical Import for Retriever Access ---
# Assume this function exists and returns a Langchain retriever interface
# You MUST implement the logic to get the actual retriever, likely in deps.py or your store module
# from research_assistant.api.deps import get_vector_store_retriever

logger = logging.getLogger(__name__)

# --- Node 1: Intent Analysis and Routing ---
def analyze_and_route_node(state: GraphState) -> dict:
    """
    Analyzes the user query, classifies intent, maintains context, performs VDB lookup,
    and determines the next step (direct answer, search, summarize, etc.).
    """
    logger.info("--- Assistant: Analyzing Query and Routing Node ---")
    query = state["query"]
    messages = state["messages"] # Conversational context
    session_id = state.get("session_id") # Need session_id for potential retriever access
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    embedding_provider = state.get("embedding_provider") # Needed for retriever

    # --- Vector DB Lookup ---
    # (Keeping the placeholder as implemented before)
    retrieved_context_str = ""
    try:
        # --- Placeholder for actual retriever implementation ---
        # Replace this with your actual VDB lookup code
        # --- Example Structure (Remove pass when implemented) ---
        # from research_assistant.api.deps import get_vector_store_retriever
        # retriever = get_vector_store_retriever(session_id=session_id, embedding_provider=embedding_provider)
        # if retriever:
        #     relevant_docs = retriever.invoke(query) # or await .ainvoke()
        #     if relevant_docs:
        #         # Format docs...
        #         retrieved_context_str = f"**Potentially Relevant Information...**\n{formatted_docs}\n"
        #         logger.info(f"Assistant: Found {len(relevant_docs)} relevant docs in Vector DB.")
        pass # Remove this line when VDB implemented
        if not retrieved_context_str:
             logger.debug("Vector DB lookup placeholder active - skipping retrieval.")
    except ImportError as ie:
        logger.warning(f"Could not import retriever function: {ie}. Skipping VDB lookup.")
    except Exception as e:
        logger.error(f"Error during Vector DB lookup: {e}", exc_info=True)
        retrieved_context_str = "" # Ensure empty on error

    # conversational history
    reversed_messages = list(reversed(messages))
    
    # Format the full history for display (newest first)
    formatted_history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in reversed_messages
    )

    # Updated System Prompt including VDB context placeholder AND History Summary Intent
    system_prompt_template = """You are the central orchestrator agent in a research assistant system. Your primary role is to analyze the **latest user query** within the **context of the conversation history** and any **potentially relevant information retrieved** to determine the most efficient path.

**Conversation History:**
{history}

{retrieved_context} {/* VDB results will be injected here (if any) */}

**Latest User Query:**
{query}

**Analyze the Latest User Query and Classify Intent (Consider retrieved context):**

1.  **Summarize Conversation History:** The user explicitly asks to summarize, review, or recall the *current* conversation. Keywords: "summarize conversation", "what did we talk about", "review our discussion", "previous conversation".
    Action: `SUMMARIZE_HISTORY`

2.  **Simple/Direct Interaction:** Greetings, farewells, thanks, clarifications, or questions **directly answered by retrieved VDB info**, general knowledge, or *very recent* history (last 1-2 turns).
    *If VDB info answers it, prefer this.*
    Action: `ANSWER_DIRECTLY`

3.  **Real-Time Data Request:** Needs current external info (news, weather, stocks). *VDB/History is unlikely sufficient.*
    Action: `NEEDS_SEARCH`

4.  **Content Summarization/Extraction Request (Not History):** Asks to summarize/extract from *external* content (e.g., search results just provided, a specific topic requiring search). *Distinguish this from summarizing the conversation itself.*
    Action: `NEEDS_CONTENT_SUMMARIZATION` (Often follows search)

5.  **Complex Task / Multi-Step Request:** Needs multiple steps (search -> summarize -> analyze). *VDB/History might inform but rarely completes it.*
    Action: `NEEDS_COMPLEX_PROCESSING`

**Output Format:**
Respond with ONLY one action keyword (`SUMMARIZE_HISTORY`, `ANSWER_DIRECTLY`, `NEEDS_SEARCH`, `NEEDS_CONTENT_SUMMARIZATION`, `NEEDS_COMPLEX_PROCESSING`) on the first line.
If `NEEDS Tesla news
Output:
NEEDS_SEARCH
latest Tesla news
"""
_SEARCH` or `NEEDS_COMPLEX_PROCESSING`, provide the concise search query on the second line.
If `NEEDS_CONTENT_SUMMARIZATION`, indicate target (e.g., "search results").

Example (History Summary):
User Query: Can you summarize our chat so far?
Output:
SUMMARIZE_HISTORY

Example (Search):
User Query: Latest
    # Format the system prompt
    formatted_system_prompt = system_prompt_template.format(
        history="{formatted_history}",
        retrieved_context=retrieved_context_str,
        query="{query}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        ("human", "{query}"),
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        # Include the latest user message in the history placeholder for the LLM call
        response = chain.invoke({"query": query})
        
        logger.debug(f"Assistant analysis raw response:\n{response}")
        lines = response.strip().split('\n')
        decision = lines[0].strip().upper()
        details = lines[1].strip() if len(lines) > 1 else ""

        search_query_formulated = query
        output_state_update = {}
        if decision == "SUMMARIZE_HISTORY": # New Decision Branch
            logger.info("Assistant Decision: Summarize History")
            output_state_update = {"next_step": "summarize_history"} # Route to summarizer node
        elif decision == "NEEDS_SEARCH":
            logger.info(f"Assistant Decision: Needs Search. Search Query: '{details or query}'")
            search_query_formulated = details if details else query
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated}
        elif decision == "ANSWER_DIRECTLY":
            logger.info("Assistant Decision: Answer Directly")
            output_state_update = {"next_step": "generate_direct_response", "retrieved_context": retrieved_context_str or None}
        # Renamed this intent for clarity
        elif decision == "NEEDS_CONTENT_SUMMARIZATION":
            logger.info(f"Assistant Decision: Needs Content Summarization (Not History). Detail: '{details}'")
            # This usually requires prior search results
            if not state.get("search_results"):
                 logger.warning("Content Summarization requested, but no search results found. Routing to search first.")
                 search_query_for_summary = query # Fallback
                 output_state_update = {"next_step": "needs_search", "search_query": search_query_for_summary}
            else:
                 # Route to the summarizer, which should prioritize search results by default now
                 output_state_update = {"next_step": "needs_summary"}
        elif decision == "NEEDS_COMPLEX_PROCESSING":
            logger.info(f"Assistant Decision: Needs Complex Processing. Initial Query Hint: '{details or query}'")
            search_query_formulated = details if details else query
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated, "complex_task": True}
        else:
             logger.warning(f"Assistant analysis failed to produce clear decision ('{decision}'), defaulting to search.")
             output_state_update = {"next_step": "needs_search", "search_query": query}

        # Add the explicit user query type for the summarizer node to use
        if decision == "SUMMARIZE_HISTORY":
            output_state_update["summary_request_type"] = "history"
        elif decision == "NEEDS_CONTENT_SUMMARIZATION":
             output_state_update["summary_request_type"] = "content"


        return output_state_update

    except Exception as e:
        logger.error(f"Error during assistant analysis LLM call: {e}", exc_info=True)
        return {"error": f"Failed to analyze query via LLM: {e}", "next_step": "needs_search", "search_query": query}


# --- Node 2: Generate Direct Response ---
# (No changes needed here)
def generate_direct_response_node(state: GraphState) -> dict:
    """Generates a response directly using the LLM. Uses VDB context if provided."""
    logger.info("--- Assistant: Generating Direct Response Node ---")
    query = state["query"]
    messages = state["messages"]
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    retrieved_context = state.get("retrieved_context")
    context_window_size = 30
    context_messages = messages[-context_window_size:]

    if retrieved_context:
        logger.info("Generating direct response using retrieved VDB context.")
        system_prompt = """You are a helpful AI assistant. The user asked a question, and relevant information was retrieved from past interactions or FAQs. Use this retrieved information *primarily* to answer the user's latest query. Also consider the recent conversation history for context. Be concise and directly address the query using the retrieved info.

**Retrieved Information (Use this first):**
{retrieved_context}

**Recent Conversation History:** (Includes the user's query as the last message)
{history}

Respond directly to the user based *primarily* on the retrieved information:"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.format(retrieved_context=retrieved_context, history="{history}")),
            MessagesPlaceholder(variable_name="history_placeholder"),
        ])
    else:
        logger.info("Generating direct response using general knowledge and history.")
        system_prompt = """You are a helpful AI assistant. Respond conversationally and directly to the user's latest query based on the provided conversation history and your general knowledge. Be concise and helpful."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history_placeholder"),
        ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"history_placeholder": context_messages})
        logger.info("Assistant generated direct response.")
        return {"final_response": response.strip()}
    except Exception as e:
        logger.error(f"Error during direct response generation: {e}", exc_info=True)
        return {"error": f"Failed to generate direct response: {e}", "final_response": "Sorry, I encountered an error trying to respond."}

# --- Node 3: Synthesize Response After Search/Summary ---
# (No changes needed here)
def synthesize_response_node(state: GraphState) -> dict:
    """Generates the final response incorporating search results and/or summary."""
    logger.info("--- Assistant: Synthesizing Response Node ---")
    query = state["query"]
    messages = state["messages"]
    search_results = state.get("search_results")
    summary = state.get("summary")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    context_window_size = 30
    context_messages_for_prompt = messages[-(context_window_size-1):-1] # History *before* latest query

    context_data = ""
    if summary:
        context_data += "Summary of Findings:\n" + summary + "\n\n"
    elif search_results:
         context_data += "Key Search Results:\n"
         formatted_results = "\n---\n".join([
            f"Title: {res.get('title', 'N/A')}\nURL: {res.get('link', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
            for res in search_results[:3]
            if isinstance(res, dict) and 'error' not in res
        ])
         context_data += formatted_results + "\n\n"

    if not context_data:
        context_data = "No specific external context was gathered or summarized for this query."
        logger.warning("Synthesizing response without search results or summary.")

    system_prompt = """You are a helpful research assistant... (rest of prompt is unchanged)""" # Keep existing prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(
            query="{query}",
            context_data="{context_data}",
            history="{history}"
            )
        ),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "query": query,
            "context_data": context_data,
            "history_placeholder": context_messages_for_prompt
            })
        logger.info("Assistant synthesized final response.")
        return {"final_response": response.strip()}
    except Exception as e:
        logger.error(f"Error during response synthesis: {e}", exc_info=True)
        return {"error": f"Failed to synthesize response: {e}", "final_response": "Sorry, I encountered an error synthesizing the information."}