import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

# Function to truncate message content
def truncate_content(content: str, max_len: int = 20) -> str:
    return content[:max_len] + "..." if len(content) > max_len else content

# --- Node 1: Intent Analysis and Routing ---
def analyze_and_route_node(state: GraphState) -> dict:
    """
    Analyzes the user query within conversation history, classifies intent,
    and determines the next step (direct answer, search, complex processing).
    """
    logger.info("--- Assistant: Analyzing Query and Routing Node ---")
    query = state["query"]
    messages = state["messages"] # Full conversational context
    provider = state.get("llm_provider")
    model = state.get("llm_model")

    # --- Format History (Truncated) ---
    # Exclude the latest message (current query) from history formatting for the prompt
    history_messages = messages[:-1] if messages else []
    # Truncate messages for the prompt context
    truncated_history = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {truncate_content(m.content)}"
        for m in history_messages if isinstance(m, BaseMessage)
    )
    
    if not truncated_history:
        truncated_history = "No previous conversation history."
    
    # --- System Prompt for Classification ---
    system_prompt_template = """You are an intent classification agent. Analyze the **Latest User Query** in the context of the **Recent Conversation History (truncated)** to determine the required action.

**Recent Conversation History (Truncated):**
{truncated_history}

**Latest User Query:**
{query}

**Classify the Latest User Query into ONE of the following categories:**

1.  **ANSWER_DIRECTLY:**
    *   Simple greetings, farewells, thanks.
    *   General knowledge questions answerable without external search (e.g., "What is 2+2?").
    *   Questions about the *immediately preceding* turns of the conversation (e.g., User asks question, AI answers, User asks "Can you explain that differently?").
    *   Explicit requests to summarize the *ongoing conversation history itself* (e.g., "Summarize what we just talked about"). -> Treat this as a direct answer task using history.

2.  **NEEDS_SEARCH:**
    *   Requires real-time information (e.g., "What's the weather today?", "Latest stock price for GOOGL?", "Current news headlines?").
    *   Requires specific, up-to-date facts or information *not* likely known by a general LLM or present in the immediate history (e.g., "Details about the latest LangChain library release?", "Who won the recent F1 race?").
    *   Requires information from external documents or URLs not already processed.

3.  **NEEDS_COMPLEX_PROCESSING:**
    *   Requires multiple steps beyond a single search query (e.g., "Compare Python and Java features.", "Analyze the pros and cons of electric cars.").
    *   Involves analysis, synthesis, comparison, evaluation, or generating a structured report based on potentially multiple information sources.
    *   Requests to summarize information *about a specific topic* that requires external search first (e.g., "Summarize the theory of relativity" - this needs search *then* summarization/synthesis).

**Output Format:**
Respond with ONLY the chosen classification keyword on the first line (`ANSWER_DIRECTLY`, `NEEDS_SEARCH`, `NEEDS_COMPLEX_PROCESSING`).
If `NEEDS_SEARCH` or `NEEDS_COMPLEX_PROCESSING`, provide a concise search query on the *second* line, suitable for a web search engine, based *only* on the **Latest User Query**.

**Example 1 (Direct Answer - History Related):**
History: Human: Tell me about LangGraph... AI: LangGraph is a fram...
Query: Can you summarize what we just discussed?
Output:
ANSWER_DIRECTLY

**Example 2 (Direct Answer - General):**
History: Human: Hi AI: Hello!
Query: What is the capital of France?
Output:
ANSWER_DIRECTLY

**Example 3 (Needs Search):**
History: Human: Hi AI: Hello!
Query: What's the weather in London right now?
Output:
NEEDS_SEARCH
weather in London now

**Example 4 (Complex Processing -> Needs Search First):**
History: (empty)
Query: Compare the advantages of solar versus wind power.
Output:
NEEDS_COMPLEX_PROCESSING
advantages solar versus wind power comparison

**Example 5 (Complex Processing -> Topic Summary -> Needs Search First):**
History: (empty)
Query: Give me a summary of the main points of the Paris Agreement.
Output:
NEEDS_COMPLEX_PROCESSING
main points Paris Agreement summary
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "Classify the latest user query based on the provided context."),
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "truncated_history": truncated_history,
            "query": query
        })

        logger.debug(f"Assistant classification raw response:\n{response}")
        lines = response.strip().split('\n')
        decision = lines[0].strip().upper()
        details = lines[1].strip() if len(lines) > 1 else "" # Potential search query

        search_query_formulated = query # Default search is the user query
        if details:
            search_query_formulated = details

        output_state_update = {}
        if decision == "ANSWER_DIRECTLY":
            logger.info("Assistant Decision: Answer Directly.")
             # Check if it's a history summary request
            if "summarize" in query.lower() and ("conversation" in query.lower() or "discussion" in query.lower() or "talked about" in query.lower() or "chat" in query.lower()):
                 logger.info("Assistant Routing: Request seems to be for History Summary.")
                 output_state_update = {"next_step": "summarize_history", "summary_request_type": "history"}
            else:
                 output_state_update = {"next_step": "generate_direct_response"}

        elif decision == "NEEDS_SEARCH":
            logger.info(f"Assistant Decision: Needs Search. Search Query: '{search_query_formulated}'")
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated, "complex_task": False} # Mark as not complex initially

        elif decision == "NEEDS_COMPLEX_PROCESSING":
            logger.info(f"Assistant Decision: Needs Complex Processing. Initial Search Query: '{search_query_formulated}'")
            # Complex tasks usually start with search
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated, "complex_task": True} # Mark as complex

        else:
            logger.warning(f"Assistant analysis failed to produce clear decision ('{decision}'), defaulting to search.")
            output_state_update = {"next_step": "needs_search", "search_query": query, "complex_task": False} # Default to simple search

        return output_state_update

    except Exception as e:
        logger.error(f"Error during assistant classification LLM call: {e}", exc_info=True)
        # Default to search on error
        return {"error": f"Failed to analyze query via LLM: {e}", "next_step": "needs_search", "search_query": query, "complex_task": False}


# --- Node 2: Generate Direct Response ---
# (No changes needed based on user request, but check system prompt)
def generate_direct_response_node(state: GraphState) -> dict:
    """Generates a response directly using the LLM. Handles general knowledge and simple history recall."""
    logger.info("--- Assistant: Generating Direct Response Node ---")
    query = state["query"]
    messages = state["messages"] # Use full messages for context here
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    # Limit context window to avoid excessive length/cost
    context_window_size = 10 # Use fewer messages for direct response context
    context_messages = messages[-context_window_size:]

    # Simple system prompt for direct answers
    system_prompt = """You are a helpful AI assistant. Answer the user's latest query directly and concisely based on the provided conversation history and your general knowledge. If the query asks to summarize the conversation, provide a brief summary of the recent messages (excluding the request itself)."""

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


# --- Node 3: Summarize History (Only for explicit history summary requests) ---
# This node is now explicitly for history summaries identified in analyze_and_route_node
def summarize_history_node(state: GraphState) -> dict:
    """Generates a summary of the conversation history."""
    logger.info("--- Assistant: Summarizing Conversation History Node ---")
    messages = state["messages"]
    provider = state.get("llm_provider")
    model = state.get("llm_model")

    # Exclude the last message (the summary request itself)
    history_to_summarize = messages[:-1]

    if not history_to_summarize:
        logger.info("No history prior to the summary request.")
        return {"final_response": "There's no prior conversation history to summarize!"}

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_to_summarize if isinstance(msg, BaseMessage)])

    system_prompt = """You are a summarization assistant. Based ONLY on the provided conversation history, create a concise summary of the main points discussed. Start the summary directly without introductory phrases like "Here is a summary...".

**Conversation History:**
{history}

**Concise Summary:**"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(history=history_str)),
        # No human message needed as history is in system prompt
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()
    try:
        summary = chain.invoke({}) # History is baked into the system prompt now
        logger.info("Assistant generated history summary.")
        # This summary is the final response for this path
        return {"final_response": summary.strip()}
    except Exception as e:
        logger.error(f"Error during history summary generation: {e}", exc_info=True)
        return {"error": f"Failed to summarize history: {e}", "final_response": "Sorry, I couldn't summarize the history due to an error."}


# --- Node 4: Synthesize Response After Search/Summary ---
# (This node handles synthesizing the FINAL response after search/external summarization)
def synthesize_response_node(state: GraphState) -> dict:
    """Generates the final response incorporating search results and/or summary of external content."""
    logger.info("--- Assistant: Synthesizing Response Node (Post-Search/Summary) ---")
    query = state["query"]
    messages = state["messages"] # Full history for final context
    search_results = state.get("search_results")
    summary = state.get("summary") # This summary comes from the summarizer_agent (for search results)
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    is_complex = state.get("complex_task", False) # Check if it was a complex task

    # Prepare context from search/summary
    context_data = ""
    if summary:
        # Summary likely came from summarizer_agent processing search results
        context_data += f"**Summary of Information Found:**\n{summary}\n\n"
        logger.info("Synthesizing response using summary of search results.")
    elif search_results and isinstance(search_results, list):
        context_data += "**Key Search Results:**\n"
        formatted_results = "\n---\n".join([
            f"Title: {res.get('title', 'N/A')}\nURL: {res.get('link', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
            for res in search_results[:5] # Limit results shown in prompt
            if isinstance(res, dict) and 'error' not in res and res.get('snippet')
        ])
        if formatted_results:
             context_data += formatted_results + "\n\n"
             logger.info("Synthesizing response using raw search results (no summary generated).")
        else:
            context_data += "No usable search results found.\n\n"
            logger.warning("Synthesizing response requested, but no valid search results or summary available.")
    else:
        context_data = "No specific external context was gathered or summarized for this query.\n\n"
        logger.warning("Synthesizing response without search results or summary.")

    # Determine the appropriate system prompt based on task complexity
    if is_complex:
        system_prompt_template = """You are a helpful research assistant. The user asked a complex question requiring analysis, comparison, or synthesis. You have been provided with relevant context (either summarized findings or raw search results).

**Original User Query:**
{query}

**Information Gathered:**
{context_data}

**Task:**
Synthesize the gathered information to provide a comprehensive answer to the user's original query. Address all parts of the query. Structure your response clearly (e.g., use bullet points for comparisons). Ensure the response is based on the provided information.

**Recent Conversation History (for context):**
{history}

**Final Answer:**"""
        logger.info("Using synthesis prompt for complex task.")
    else: # Simple search result presentation
         system_prompt_template = """You are a helpful research assistant. The user asked a question that required a web search. You have been provided with relevant context (either summarized findings or raw search results).

**Original User Query:**
{query}

**Information Found:**
{context_data}

**Task:**
Answer the user's query directly using the information found. Be concise and focus on the key findings relevant to the query. If summarizing results, state that clearly.

**Recent Conversation History (for context):**
{history}

**Final Answer:**"""
         logger.info("Using synthesis prompt for simple search task.")

    # Format history for the synthesis prompt (limit length)
    context_window_size = 10
    history_messages_for_prompt = messages[-context_window_size:]
    formatted_history = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in history_messages_for_prompt if isinstance(m, BaseMessage)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template.format(
            query=query,
            context_data=context_data,
            history=formatted_history
            )
        ),
        # No explicit human message placeholder needed if query is in system prompt
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({}) # All context is in the system prompt
        logger.info("Assistant synthesized final response.")
        return {"final_response": response.strip()}
    except Exception as e:
        logger.error(f"Error during response synthesis: {e}", exc_info=True)
        return {"error": f"Failed to synthesize response: {e}", "final_response": "Sorry, I encountered an error synthesizing the information."}