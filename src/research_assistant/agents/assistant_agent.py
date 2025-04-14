import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- Node 1: Intent Analysis and Routing ---
def analyze_and_route_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user query, classifies intent, maintains context, performs VDB lookup,
    and determines the next step (direct answer, search, summarize, etc.).
    """
    logger.info("--- Assistant: Analyzing Query and Routing Node ---")
    query = state["query"]
    messages = state["messages"]
    session_id = state.get("session_id")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    embedding_provider = state.get("embedding_provider")

    # --- Vector DB Lookup --- 
    retrieved_context_str = ""
    try:
        # Placeholder for actual retriever implementation
        logger.debug("Vector DB lookup placeholder active - skipping retrieval.")
    except Exception as e:
        logger.error(f"Error during Vector DB lookup for session {session_id}: {e}", exc_info=True)
        retrieved_context_str = ""

    # Limit conversational history 
    context_window_size = 30
    context_messages = messages[-context_window_size:]

    system_prompt_template = """You are the central orchestrator agent in a research assistant system. Your primary role is to analyze the **latest user query** within the **context of the conversation history** and any **potentially relevant information retrieved from past interactions or FAQs** to determine the most efficient path to fulfill the request.

**Conversation History:**
{history}

{retrieved_context}

**Latest User Query:**
{query}

**Analyze the Latest User Query and Classify Intent (Consider retrieved context):**

1.  **Simple/Direct Interaction:** Greetings, farewells, thanks, clarifications, or questions **directly answered by the 'Potentially Relevant Information' section (if provided)**, general knowledge, or recent history.
    *If retrieved information provides a sufficient answer, prefer this.* 
    Action: `ANSWER_DIRECTLY`

2.  **Real-Time Data Request:** Needs current external info (news, weather, stocks). *Retrieved information is unlikely sufficient.*
    Action: `NEEDS_SEARCH`

3.  **Content Summarization/Extraction Request:** Asks to summarize/extract from *previous* messages or search results. *Check if retrieved information fulfills this, otherwise proceed.*
    Action: `NEEDS_SUMMARIZATION` (Often follows search)

4.  **Complex Task / Multi-Step Request:** Needs multiple steps (search -> summarize -> analyze). *Retrieved information might inform the task but rarely completes it.*
    Action: `NEEDS_COMPLEX_PROCESSING`

**Output Format:**
Respond with ONLY one action keyword (`ANSWER_DIRECTLY`, `NEEDS_SEARCH`, `NEEDS_SUMMARIZATION`, `NEEDS_COMPLEX_PROCESSING`) on the first line.
If `NEEDS_SEARCH` or `NEEDS_COMPLEX_PROCESSING`, provide the concise search query on the second line.
If `NEEDS_SUMMARIZATION`, indicate target (e.g., "search results")."""

    formatted_system_prompt = system_prompt_template.format(
        history="{history}",
        retrieved_context=retrieved_context_str,
        query="{query}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])

    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "history_placeholder": context_messages,
            "query": query 
        })
        
        logger.debug(f"Assistant analysis raw response:\n{response}")
        lines = response.strip().split('\n')
        decision = lines[0].strip().upper()
        details = lines[1].strip() if len(lines) > 1 else ""

        output_state_update = {}

        if decision == "NEEDS_SEARCH":
            search_query_formulated = details if details else query
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated}
        elif decision == "ANSWER_DIRECTLY":
            output_state_update = {"next_step": "generate_direct_response", "retrieved_context": retrieved_context_str or None}
        elif decision == "NEEDS_SUMMARIZATION":
            if not state.get("search_results"):
                output_state_update = {"next_step": "needs_search", "search_query": query}
            else:
                output_state_update = {"next_step": "needs_summary"}
        elif decision == "NEEDS_COMPLEX_PROCESSING":
            search_query_formulated = details if details else query
            output_state_update = {"next_step": "needs_search", "search_query": search_query_formulated, "complex_task": True}
        else:
            output_state_update = {"next_step": "needs_search", "search_query": query}
        
        return output_state_update

    except Exception as e:
        logger.error(f"Error during assistant analysis LLM call: {e}", exc_info=True)
        return {"error": f"Failed to analyze query via LLM: {e}", "next_step": "needs_search", "search_query": query}

# --- Node 2: Generate Direct Response ---
def generate_direct_response_node(state: GraphState) -> Dict[str, Any]:
    """Generates a response directly using the LLM."""
    logger.info("--- Assistant: Generating Direct Response Node ---")
    query = state["query"]
    messages = state["messages"]
    provider = state.get("llm_provider", "openai")
    model = state.get("llm_model")
    retrieved_context = state.get("retrieved_context")

    context_window_size = 30 
    context_messages = messages[-context_window_size:]

    if retrieved_context:
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
def synthesize_response_node(state: GraphState) -> Dict[str, Any]:
    """Generates the final response incorporating search results and/or summary."""
    logger.info("--- Assistant: Synthesizing Response Node ---")
    query = state["query"]
    messages = state["messages"]
    search_results = state.get("search_results")
    summary = state.get("summary")
    provider = state.get("llm_provider", "openai")
    model = state.get("llm_model")

    context_window_size = 30 
    context_messages_for_prompt = messages[-(context_window_size-1):-1]

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

    system_prompt = """You are a helpful research assistant. Your task is to synthesize the provided information (summary and/or key search results) to answer the user's *original query* comprehensively, clearly, and conversationally.

    **Conversation History (for context):**
    {history}

    **User's Original Query:**
    {query}

    **Information Gathered (from Search/Summary):**
    {context_data}

    **Instructions:**
    1.  **Directly address** the user's original query using the 'Information Gathered'.
    2.  **Cite sources** implicitly or explicitly if appropriate.
    3.  **Frame the response** contextually.
    4.  If the gathered information is insufficient to fully answer, acknowledge that but provide the best possible answer based on what's available.
    5.  **Maintain a helpful and conversational tone.**
    6.  **(Optional) Suggest relevant follow-up questions** or actions the user might be interested in based on the topic and findings."""

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