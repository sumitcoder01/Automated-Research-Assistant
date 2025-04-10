import logging
from research_assistant.llms.provider import get_llm
from research_assistant.assistant.graph.state import GraphState
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

logger = logging.getLogger(__name__)

# --- Node 1: Initial Analysis and Routing ---
def analyze_query_node(state: GraphState) -> dict:
    """Analyzes query, decides if direct answer or search is needed."""
    logger.info("--- Assistant: Analyzing Query Node ---")
    query = state["query"]
    messages = state["messages"] # History for context
    provider = state.get("llm_provider")
    model = state.get("llm_model")

    # System prompt to guide the LLM's decision
    system_prompt = """You are the supervisor of a research assistant system. Your job is to analyze the user's query and the conversation history.
    Determine the best course of action:
    1.  **Answer Directly:** If the query is conversational (e.g., greeting) or requests information you likely already know or can infer from the history.
    2.  **Needs Search:** If the query requires real-time information, specific research data (like papers), or knowledge beyond your internal capabilities.

    Based on your decision, respond with ONLY one of the following words:
    `ANSWER_DIRECTLY`
    `NEEDS_SEARCH`

    If you decide NEEDS_SEARCH, also formulate the best possible, concise search query based on the user's request and conversation history on the next line. Example:
    NEEDS_SEARCH
    latest advancements in AI regulation Europe
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # Add previous messages for context - keep it concise
        *messages[-4:-1], # Exclude the *very* last user message if it's the current query
        ("human", "User Query: {query}")
    ])
    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query})
        logger.debug(f"Assistant analysis response: {response}")
        lines = response.strip().split('\n')
        decision = lines[0].strip().upper()
        search_query_formulated = lines[1].strip() if len(lines) > 1 else query # Fallback

        if decision == "NEEDS_SEARCH":
            logger.info("Assistant Decision: Needs Search")
            return {"next_step": "needs_search", "search_query": search_query_formulated}
        elif decision == "ANSWER_DIRECTLY":
            logger.info("Assistant Decision: Answer Directly")
            return {"next_step": "generate_direct_response"}
        else:
             # Fallback if LLM doesn't follow instructions
             logger.warning(f"Assistant analysis failed to produce clear decision ('{decision}'), defaulting to search.")
             return {"next_step": "needs_search", "search_query": query} # Fallback to search

    except Exception as e:
        logger.error(f"Error during assistant analysis: {e}", exc_info=True)
        return {"error": f"Failed to analyze query: {e}", "next_step": "needs_search", "search_query": query} # Fallback

# --- Node 2: Generate Direct Response ---
def generate_direct_response_node(state: GraphState) -> dict:
    """Generates a response directly using the LLM (no search)."""
    logger.info("--- Assistant: Generating Direct Response Node ---")
    query = state["query"]
    messages = state["messages"]
    provider = state.get("llm_provider", "openai")
    model = state.get("llm_model")

    # Simple prompt for direct answers/conversation
    system_prompt = """You are a helpful AI assistant. Respond conversationally to the user's query based on the context provided."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        *messages[:-1], # Include history
        ("human", "{query}")
    ])
    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query})
        logger.info("Assistant generated direct response.")
        # Append user query and AI response to messages for next turn? Decide based on state management needs.
        # For now, just return the final response
        return {"final_response": response.strip()}
    except Exception as e:
        logger.error(f"Error during direct response generation: {e}", exc_info=True)
        return {"error": f"Failed to generate direct response: {e}", "final_response": "Sorry, I encountered an error trying to respond."}

# --- Node 3: Synthesize Response After Search/Summary ---
def synthesize_response_node(state: GraphState) -> dict:
    """Generates the final response incorporating search results and/or summary."""
    logger.info("--- Assistant: Synthesizing Response Node ---")
    query = state["query"]
    messages = state["messages"]
    search_results = state.get("search_results")
    summary = state.get("summary")
    provider = state.get("llm_provider", "openai")
    model = state.get("llm_model")

    # Build context string
    context = ""
    if summary:
        context += "Summary of Findings:\n" + summary + "\n\n"
    elif search_results:
         # Format some search results if no summary available
         context += "Key Search Results:\n"
         formatted_results = "\n---\n".join([
            f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
            for res in search_results[:3] # Limit context length
            if isinstance(res, dict) and 'error' not in res
        ])
         context += formatted_results + "\n\n"

    system_prompt = """You are a helpful research assistant. Synthesize the provided information (summary and/or search results) to answer the user's original query comprehensively and clearly. Respond directly to the user. If the context is insufficient, state that but provide the best possible answer based on the available information."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        *messages[:-1], # History context
         ("human", "User's Original Query: {query}\n\nAvailable Context:\n{context}\n\nAnswer:")
    ])
    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query, "context": context or "No specific context was gathered."})
        logger.info("Assistant synthesized final response.")
        return {"final_response": response.strip()}
    except Exception as e:
        logger.error(f"Error during response synthesis: {e}", exc_info=True)
        return {"error": f"Failed to synthesize response: {e}", "final_response": "Sorry, I encountered an error synthesizing the information."}