import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser

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

    # conversational history
    reversed_messages = list(reversed(messages))
    
    # Format the full history for display (newest first)
    formatted_history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in reversed_messages
    )
    
 # --- System Prompt Using Query + History Context ---
    system_prompt_template = """You are the central orchestrator agent in a research assistant system. Your primary role is to analyze the **latest user query** within the context of the **recent conversation history** to determine the most efficient path. Do NOT use external data for this initial classification. Prioritize identifying requests to summarize the conversation history itself.

**Recent Conversation History:**
{history}

**Latest User Query:** (Already included in the history placeholder below)
{query}

**Analyze the Latest User Query (in context of history) and Classify Intent:**

1.  **Conversation History Summary Request:** **(PRIORITY)** The user *explicitly* asks to summarize, review, or recall the *ongoing chat or past conversation itself*. Keywords: "summarize **this conversation**", "**what did we talk about**", "**review our discussion**", "**remind me what we said about**...", "can you give me a summary of **our chat**?", "**summarize the previous conversation**". If the latest query matches this pattern, choose this action.
    Action: `SUMMARIZE_HISTORY`

2.  **Simple/Direct Interaction:** The query is a basic greeting ("Hi", "Hello"), farewell ("Bye"), thanks ("Thank you"), a very simple clarification request related *only* to the immediate flow (e.g., "What was that?", "Can you repeat?"), OR a simple follow-up question that can likely be answered from the *immediately preceding* turns in the history.
    Action: `ANSWER_DIRECTLY`

3.  **Data Request / Question:** The query asks a factual question that requires external information (news, facts, specific data - e.g., "What is LangChain?", "Who won the election?") OR asks a question that cannot be answered from general knowledge or the provided history and isn't a request to summarize the *conversation* (Rule 1).
    Action: `NEEDS_SEARCH`

4.  **Topic Summarization/Extraction Request:** The user asks to summarize or extract information *about a specific topic, subject, or concept* (e.g., "summarize the plot of Hamlet", "extract the main points about renewable energy", "tell me the key features of LangGraph"). This requires gathering external information first. **This is NOT Rule 1 (Summarizing the conversation).**
    Action: `NEEDS_SEARCH` (The search results will be summarized later if needed)

5.  **Complex Task / Multi-Step Request:** The query implies multiple steps beyond a simple search or topic summary (e.g., "compare Python and Java", "analyze the pros and cons of electric cars", "research and write a short report on AI ethics").
    Action: `NEEDS_COMPLEX_PROCESSING`

**Output Format:**
Respond with ONLY one action keyword (`SUMMARIZE_HISTORY`, `ANSWER_DIRECTLY`, `NEEDS_SEARCH`, `NEEDS_COMPLEX_PROCESSING`) on the first line.
If `NEEDS_SEARCH` or `NEEDS_COMPLEX_PROCESSING`, provide a concise search query on the second line (often based on the latest user query).

Example (Conversation History Summary):
History: User: Tell me about LangGraph. Assistant: [Explains LangGraph]. User: Please summarize our conversation.
Latest User Query: Please summarize our conversation.
Output:
SUMMARIZE_HISTORY

Example (Simple Follow-up -> Direct Answer):
History: User: What is LangGraph? Assistant: It's a framework for building stateful LLM applications. User: Okay, thanks.
Latest User Query: Okay, thanks.
Output:
ANSWER_DIRECTLY

Example (Topic Summary -> Needs Search):
History: User: Hi Assistant: Hello! User: Can you summarize the theory of relativity?
Latest User Query: Can you summarize the theory of relativity?
Output:
NEEDS_SEARCH
theory of relativity summary
"""

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